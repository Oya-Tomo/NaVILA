"""Run NaVILA inference as a safe Zenoh-controlled Go2 node."""

from __future__ import annotations

import argparse
import copy
import logging
import math
import signal
import time
import warnings
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from io import BytesIO
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from types import FrameType
from typing import Any, Callable

from config import (
    JSON_ENCODING,
    HeartbeatCommand,
    NodeConfig,
    NodeState,
    PostureCommand,
    VelocityCommand,
    decode_control_command,
    decode_go2_state,
    load_node_config,
)
from controller import ZERO_VELOCITY, Controller, InferenceRequest
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

import zenoh

DEFAULT_NODE_CONFIG_PATH = Path("zenoh/node-config.json5")
DEFAULT_ZENOH_CONFIG_PATH = Path("zenoh/zenoh-config.json5")
WORKER_POLL_SECONDS = 0.05
WORKER_JOIN_SECONDS = 1.0
LOGGER = logging.getLogger(__name__)
EVAL_PADDING_SIZE = (512, 512)
MODEL_SYSTEM_PROMPT = (
    "You are a helpful language and vision assistant. You are able to understand the visual content "
    "that the user provides, and assist the user with a variety of tasks using natural language."
)
_GREEDY_GENERATION_OPTIONS: dict[str, bool | float | int | None] = {
    "do_sample": False,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 50,
    "typical_p": 1.0,
    "epsilon_cutoff": 0.0,
    "eta_cutoff": 0.0,
    "penalty_alpha": None,
    "num_beams": 1,
}
_KNOWN_MODEL_LOG_PREFIXES = {
    "bitsandbytes.cextension": "WARNING: BNB_CUDA_VERSION=130 environment variable detected",
    "transformers.tokenization_utils_base": "Special tokens have been added in the vocabulary",
}


class _KnownModelAdvisoryFilter(logging.Filter):
    """Hide only advisories already accounted for by the pinned Jetson stack."""

    def filter(self, record: logging.LogRecord) -> bool:
        prefix = _KNOWN_MODEL_LOG_PREFIXES.get(record.name)
        return prefix is None or not record.getMessage().startswith(prefix)


@contextmanager
def _suppress_known_model_advisories() -> Iterator[None]:
    """Scope known upstream warning noise to model loading and generation."""

    log_filter = _KnownModelAdvisoryFilter()
    filtered_loggers = [logging.getLogger(name) for name in _KNOWN_MODEL_LOG_PREFIXES]
    for logger in filtered_loggers:
        logger.addFilter(log_filter)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`resume_download` is deprecated and will be removed in version 1\.0\.0\.",
                category=FutureWarning,
                module=r"huggingface_hub\.file_download",
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Found GPU\d+ Orin which is of compute capability \(CC\) 8\.7\.",
                category=UserWarning,
                module=r"torch\.cuda",
            )
            warnings.filterwarnings(
                "ignore",
                message=r"_check_is_size will be removed in a future PyTorch release.*",
                category=FutureWarning,
                module=r"bitsandbytes\.(?:backends\.cuda\.ops|_ops)",
            )
            yield
    finally:
        for logger in filtered_loggers:
            logger.removeFilter(log_filter)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="navila-zenoh-node",
        description="Run camera-driven NaVILA inference and publish safe Go2 commands.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--node-config",
        type=Path,
        default=DEFAULT_NODE_CONFIG_PATH,
        metavar="FILE",
        help="NaVILA node JSON5 configuration file.",
    )
    parser.add_argument(
        "--zenoh-config",
        type=Path,
        default=DEFAULT_ZENOH_CONFIG_PATH,
        metavar="FILE",
        help="Zenoh JSON5 configuration file.",
    )
    return parser


def next_future_deadline(deadline: float, period: float, now: float) -> float:
    """Advance an absolute schedule without replaying missed slots."""

    if now < deadline:
        return deadline
    missed = math.floor((now - deadline) / period) + 1
    return deadline + missed * period


def prepare_inference_frames(
    frames: Sequence[Image.Image],
    num_video_frames: int,
) -> tuple[Image.Image, ...]:
    """Keep the latest frames and apply the evaluation path's black left padding."""

    if not frames:
        raise ValueError("inference requires at least one real camera frame")
    prepared = list(frames[-num_video_frames:])
    padding = [Image.new("RGB", EVAL_PADDING_SIZE, color=(0, 0, 0)) for _ in range(num_video_frames - len(prepared))]
    return tuple(padding + prepared)


def build_navigation_question(instruction: str, frame_count: int) -> str:
    historical_tokens = "<image>\n" * (frame_count - 1)
    return (
        "Imagine you are a robot programmed for navigation tasks. You have been given a video "
        f"of historical observations {historical_tokens}, and current observation <image>\n. "
        f'Your assigned task is: "{instruction}" '
        "Analyze this series of images to decide your next action, which could be turning left or right by a "
        "specific degree, moving forward a certain distance, or stop if the task is completed."
    )


def build_model_prompt(tokenizer: Any, question: str) -> str:
    """Render navigation text with the chat template shipped by the checkpoint."""

    messages = [
        {"role": "system", "content": MODEL_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("loaded tokenizer does not provide a usable chat template") from error
    if not isinstance(prompt, str) or not prompt:
        raise RuntimeError("loaded tokenizer produced an invalid chat prompt")
    return prompt


def build_generation_config(model: Any, tokenizer: Any) -> Any:
    """Copy model-specific token settings and normalize deterministic decoding."""

    try:
        generation_config = copy.deepcopy(model.llm.generation_config)
    except AttributeError as error:
        raise RuntimeError("loaded model does not provide an LLM generation config") from error

    for name, value in _GREEDY_GENERATION_OPTIONS.items():
        setattr(generation_config, name, value)
    generation_config.max_new_tokens = 32
    generation_config.use_cache = True

    if getattr(generation_config, "eos_token_id", None) is None:
        generation_config.eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if generation_config.eos_token_id is None:
        raise RuntimeError("loaded model and tokenizer do not provide an EOS token")

    if getattr(generation_config, "pad_token_id", None) is None:
        generation_config.pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if generation_config.pad_token_id is None:
        eos_token_ids = generation_config.eos_token_id
        generation_config.pad_token_id = eos_token_ids[0] if isinstance(eos_token_ids, (list, tuple)) else eos_token_ids

    return generation_config


def _json_payload(model: Any) -> str:
    return model.json(separators=(",", ":"))


def _has_encoding(sample: Any, expected: str) -> bool:
    encoding = str(getattr(sample, "encoding", ""))
    return encoding == expected or encoding.startswith(f"{expected};")


@contextmanager
def stop_on_signals() -> Iterator[Event]:
    """Convert SIGINT/SIGTERM into a cooperative shutdown request."""

    requested = Event()
    previous_handlers: dict[int, Any] = {}

    def request_stop(_signum: int, _frame: FrameType | None) -> None:
        requested.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)
    try:
        yield requested
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


class NaVILAEngine:
    """Thin adapter around the existing NaVILA model-loading and eval path."""

    def __init__(self, tokenizer: Any, model: Any, image_processor: Any, *, num_video_frames: int) -> None:
        if isinstance(num_video_frames, bool) or not isinstance(num_video_frames, int) or num_video_frames <= 0:
            raise ValueError("model.config.num_video_frames must be a positive integer")
        self._tokenizer = tokenizer
        self._model = model
        self._image_processor = image_processor
        self.num_video_frames = num_video_frames

    @classmethod
    def load(cls, config: NodeConfig) -> NaVILAEngine:
        with _suppress_known_model_advisories():
            import torch

            from llava.mm_utils import get_model_name_from_path
            from llava.model.builder import load_pretrained_model

            quantization = config.model.quantization
            tokenizer, model, image_processor, _context_length = load_pretrained_model(
                model_path=config.model.key,
                model_base=None,
                model_name=get_model_name_from_path(config.model.key),
                load_8bit=quantization == "8bit",
                load_4bit=quantization == "4bit",
                device_map="auto",
                device="cuda",
                torch_dtype=torch.float16,
            )
        num_video_frames = getattr(model.config, "num_video_frames", None)
        if image_processor is None:
            raise RuntimeError("loaded model does not provide an image processor")
        return cls(tokenizer, model, image_processor, num_video_frames=num_video_frames)

    def infer(self, request: InferenceRequest) -> str:
        import torch

        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import process_images, tokenizer_image_token

        frames = prepare_inference_frames(request.frames, self.num_video_frames)
        question = build_navigation_question(request.instruction, len(frames))
        prompt = build_model_prompt(self._tokenizer, question)

        device = self._model.device
        images = process_images(frames, self._image_processor, self._model.config)
        if isinstance(images, list):
            images = [image.to(device=device, dtype=torch.float16) for image in images]
        else:
            images = images.to(device=device, dtype=torch.float16)
        input_ids = (
            tokenizer_image_token(
                prompt,
                self._tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            )
            .unsqueeze(0)
            .to(device)
        )

        generation_config = build_generation_config(self._model, self._tokenizer)
        with _suppress_known_model_advisories(), torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                images=images,
                generation_config=generation_config,
            )
        return self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


class NodeRuntime:
    """Own one Zenoh session, its callbacks, and the two node workers."""

    def __init__(
        self,
        config: NodeConfig,
        engine: NaVILAEngine,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._engine = engine
        self._clock = clock
        self.controller = Controller(
            config,
            num_video_frames=engine.num_video_frames,
            clock=clock,
        )
        self._stop_workers = Event()
        self._wake_control = Event()
        self._worker_failures: Queue[tuple[str, Exception]] = Queue()
        self._camera_decode_lock = Lock()
        self._velocity_publisher: Any = None
        self._posture_publisher: Any = None
        self._state_publisher: Any = None

    def run(self, zenoh_config: zenoh.Config, shutdown_requested: Event) -> None:
        zenoh.init_log_from_env_or("error")
        keyspace = self._keyspace()
        worker_failure: tuple[str, Exception] | None = None

        with zenoh.open(zenoh_config) as session, ExitStack() as resources:
            self._velocity_publisher = resources.enter_context(
                session.declare_publisher(
                    keyspace["go2_command"],
                    encoding=JSON_ENCODING,
                    congestion_control=zenoh.CongestionControl.DROP,
                    reliability=zenoh.Reliability.BEST_EFFORT,
                )
            )
            self._posture_publisher = resources.enter_context(
                session.declare_publisher(
                    keyspace["go2_command"],
                    encoding=JSON_ENCODING,
                    congestion_control=zenoh.CongestionControl.BLOCK,
                    reliability=zenoh.Reliability.RELIABLE,
                )
            )
            self._state_publisher = resources.enter_context(
                session.declare_publisher(
                    keyspace["state"],
                    encoding=JSON_ENCODING,
                    congestion_control=zenoh.CongestionControl.DROP,
                    reliability=zenoh.Reliability.BEST_EFFORT,
                )
            )
            resources.enter_context(session.declare_subscriber(keyspace["command"], self._on_control))
            resources.enter_context(session.declare_subscriber(keyspace["camera"], self._on_camera))
            resources.enter_context(session.declare_subscriber(keyspace["go2_state"], self._on_go2_state))

            control_worker = Thread(target=self._run_control_worker, name="navila-control", daemon=True)
            inference_worker = Thread(target=self._run_inference_worker, name="navila-inference", daemon=True)
            control_worker.start()
            self.controller.initialize()
            self._wake_control.set()
            inference_worker.start()

            try:
                while not shutdown_requested.wait(WORKER_POLL_SECONDS):
                    try:
                        worker_failure = self._worker_failures.get_nowait()
                    except Empty:
                        continue
                    break
            finally:
                self._shutdown(control_worker, inference_worker)

            if worker_failure is None:
                try:
                    worker_failure = self._worker_failures.get_nowait()
                except Empty:
                    pass

        if worker_failure is not None:
            worker, error = worker_failure
            raise RuntimeError(f"{worker} worker failed: {error}") from error
        shutdown_error = self.controller.shutdown_error
        if shutdown_error is not None:
            raise RuntimeError(shutdown_error)

    def _keyspace(self) -> dict[str, str]:
        return {
            "camera": self._config.camera.key,
            "command": f"{self._config.zenoh_key_prefix}/command",
            "state": f"{self._config.zenoh_key_prefix}/state",
            "go2_command": f"{self._config.go2.zenoh_key_prefix}/command",
            "go2_state": f"{self._config.go2.zenoh_key_prefix}/state",
        }

    def _on_control(self, sample: Any) -> None:
        try:
            if not _has_encoding(sample, JSON_ENCODING):
                raise ValueError(f"expected {JSON_ENCODING}, got {sample.encoding}")
            command = decode_control_command(sample.payload.to_bytes())
        except (UnicodeDecodeError, ValidationError, ValueError) as error:
            LOGGER.warning("Rejected invalid NaVILA control command: %s", error)
            return
        accepted = self.controller.handle_command(command, received_at=self._clock())
        if not accepted and not isinstance(command, HeartbeatCommand):
            LOGGER.info("Ignored %s command in the current state", command.type)
        self._wake_control.set()

    def _on_go2_state(self, sample: Any) -> None:
        try:
            if not _has_encoding(sample, JSON_ENCODING):
                raise ValueError(f"expected {JSON_ENCODING}, got {sample.encoding}")
            state = decode_go2_state(sample.payload.to_bytes())
        except (UnicodeDecodeError, ValidationError, ValueError) as error:
            LOGGER.warning("Rejected invalid Go2 state: %s", error)
            return
        self.controller.update_go2_state(state, received_at=self._clock())
        self._wake_control.set()

    def _on_camera(self, sample: Any) -> None:
        if not _has_encoding(sample, "image/jpeg"):
            LOGGER.warning("Rejected camera sample with encoding %s", getattr(sample, "encoding", None))
            return
        now = self._clock()
        if not self.controller.camera_sample_due(now) or not self._camera_decode_lock.acquire(blocking=False):
            return
        try:
            if not self.controller.camera_sample_due(self._clock()):
                return
            with Image.open(BytesIO(sample.payload.to_bytes())) as source:
                frame = source.convert("RGB")
                frame.load()
            if self.controller.append_camera_frame(frame, sampled_at=self._clock()):
                self._wake_control.set()
        except (OSError, UnidentifiedImageError, ValueError) as error:
            LOGGER.warning("Rejected invalid camera JPEG: %s", error)
        finally:
            self._camera_decode_lock.release()

    def _run_control_worker(self) -> None:
        state_period = 1.0 / self._config.node_state_publish_frequency_hz
        velocity_period = 1.0 / self._config.go2_velocity_publish_frequency_hz
        next_state = self._clock()
        next_velocity = next_state
        try:
            while not self._stop_workers.is_set():
                now = self._clock()
                self.controller.advance(now=now)
                effects = self.controller.take_effects()
                if effects.zero_velocity:
                    self._publish_velocity(ZERO_VELOCITY)
                if effects.posture is not None:
                    self._posture_publisher.put(_json_payload(PostureCommand(posture=effects.posture)))

                now = self._clock()
                if now >= next_velocity:
                    velocity = self.controller.velocity_for_publish(now=now)
                    if velocity is not None:
                        self._publish_velocity(velocity)
                    next_velocity = next_future_deadline(next_velocity, velocity_period, now)
                if now >= next_state:
                    self._publish_state(self.controller.snapshot(now=now))
                    next_state = next_future_deadline(next_state, state_period, now)

                timeout = max(0.0, min(next_state, next_velocity, now + WORKER_POLL_SECONDS) - self._clock())
                self._wake_control.wait(timeout)
                self._wake_control.clear()
        except Exception as error:
            self._worker_failures.put(("control", error))

    def _run_inference_worker(self) -> None:
        period = 1.0 / self._config.inference.frequency_hz
        next_inference = self._clock()
        try:
            while not self._stop_workers.is_set():
                delay = next_inference - self._clock()
                if delay > 0 and self._stop_workers.wait(delay):
                    return
                request = self.controller.begin_inference(now=self._clock())
                if request is not None:
                    started_at = self._clock()
                    try:
                        output = self._engine.infer(request)
                    except Exception as error:
                        LOGGER.exception("NaVILA inference failed")
                        self.controller.fail_inference(error, failed_at=self._clock())
                    else:
                        completed_at = self._clock()
                        duration_sec = completed_at - started_at
                        LOGGER.info("NaVILA inference output (%.3fs): %s", duration_sec, output)
                        self.controller.complete_inference(
                            output,
                            duration_sec=duration_sec,
                            completed_at=completed_at,
                        )
                    self._wake_control.set()
                next_inference = next_future_deadline(next_inference, period, self._clock())
        except Exception as error:
            self._worker_failures.put(("inference", error))

    def _publish_velocity(self, command: VelocityCommand) -> None:
        self._velocity_publisher.put(_json_payload(command))

    def _publish_state(self, state: NodeState) -> None:
        self._state_publisher.put(_json_payload(state))

    def _shutdown(self, control_worker: Thread, inference_worker: Thread) -> None:
        self.controller.request_shutdown(now=self._clock())
        self._wake_control.set()
        deadline = self._clock() + self._config.go2.down_timeout_seconds
        while not self.controller.shutdown_complete and control_worker.is_alive() and self._clock() < deadline:
            self._wake_control.set()
            time.sleep(WORKER_POLL_SECONDS)

        self._stop_workers.set()
        self._wake_control.set()
        control_worker.join(WORKER_JOIN_SECONDS)
        inference_worker.join(WORKER_JOIN_SECONDS)
        if control_worker.is_alive():
            raise RuntimeError("control worker did not stop")
        if inference_worker.is_alive():
            raise RuntimeError("inference worker did not stop before shutdown deadline")
        if not self.controller.shutdown_complete:
            raise RuntimeError(
                "shutdown timed out before Go2 resting state and inference completion"
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        node_config = load_node_config(args.node_config)
        zenoh_config = zenoh.Config.from_file(args.zenoh_config)
        with stop_on_signals() as shutdown_requested:
            engine = NaVILAEngine.load(node_config)
            NodeRuntime(node_config, engine).run(zenoh_config, shutdown_requested)
    except KeyboardInterrupt:
        LOGGER.info("Stopped")
    except (OSError, RuntimeError, ValueError, ValidationError, zenoh.ZError) as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
