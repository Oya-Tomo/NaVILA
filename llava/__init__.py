import os

# bitsandbytes 0.49.x ships no prebuilt binary for CUDA 13.2 (only up to
# cuda130). On a CUDA 13.2 stack (e.g. JetPack 7.2 / torch cu132) the NF4
# 4-bit path otherwise fails to load libbitsandbytes_cuda132.so. CUDA 13.0
# is minor-version compatible, so transparently select it. bitsandbytes
# upstream CUDA 13.2 support is tracked in #1937; revisit once a cuda132
# wheel is released (and drop this workaround).
try:
    import torch as _torch
    if str(getattr(_torch.version, "cuda", "") or "").startswith("13.2"):
        os.environ.setdefault("BNB_CUDA_VERSION", "130")
except Exception:
    pass

from .entry import *
from .media import *
