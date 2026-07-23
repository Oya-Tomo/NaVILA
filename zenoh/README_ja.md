# NaVILA Zenoh ノード

[English](README.md)

このディレクトリは、互いに独立した2つのプロセスを提供します。

- `node.py` はカメラ JPEG を受信し、NaVILA の観測履歴を保持して推論を実行し、現在の結果を安全に Go2 コマンドへ変換します。
- `cli.py` は instruction、`start`、`stop`、heartbeat を NaVILA ノードへ送ります。Go2 のキーへ直接アクセスすることはありません。

カメラとロボットのブリッジは、別プロジェクトかつ別プロセスです。

- [camera-zenoh-node](https://github.com/Oya-Tomo/camera-zenoh-node)
- [unitree-go2-zenoh-node](https://github.com/Oya-Tomo/unitree-go2-zenoh-node)

これらは submodule でも vendoring された依存でもありません。このリポジトリから clone、install、起動、停止、監視は行いません。それぞれを別の checkout に用意し、各プロジェクトの説明に従ってセットアップしてください。

> [!WARNING]
> このソフトウェアは実機を動かします。まず録画済み JPEG とロボットなしの構成で試してください。実機試験では Go2 を支持し、低速設定を使い、リモコンを手元に置き、緊急停止手順を事前に決めてください。`unitree/go2/command` へ別 publisher から同時に送信する運用は非対応ですが、技術的には禁止していません。

## データフローと責務

```text
camera-zenoh-node               NaVILA node                     unitree-go2-zenoh-node
camera/front (JPEG) ----------> frame history
                                inference/current action ------> unitree/go2/command
                         +----> lifecycle/state <--------------- unitree/go2/state
                         |
operator <----> CLI -----+
             navila/command
             navila/state
```

NaVILA ノードはモデル推論と、高水準の `stand`、速度、zero、`down` 要求を担当します。Go2 ノードは観測状態に基づく姿勢遷移を担当し、`StopMove`、静止確認、`StandDown` を実行します。CLI はオペレータ操作だけを担当します。

分散システムとして厳密な起動順はありません。カメラ、Go2 ノード、CLI がなくても NaVILA ノードは idle のまま動作します。ただし最初から readiness を確認できるよう、カメラと Go2 ノード、NaVILA ノード、CLI の順を推奨します。

## セットアップ

NaVILA リポジトリのルートで実行します。

```console
$ uv sync --group zenoh
$ cp zenoh/node-config.example.json5 zenoh/node-config.json5
$ cp zenoh/zenoh-config.example.json5 zenoh/zenoh-config.json5
$ cp zenoh/cli-config.example.json5 zenoh/cli-config.json5
$ cp zenoh/cli-zenoh-config.example.json5 zenoh/cli-zenoh-config.json5
```

4つの runtime file をすべて編集してください。`node-config.json5` と `cli-config.json5` は Pydantic で厳密に検証します。2つの `*-zenoh-config.json5` は Zenoh に直接渡し、mode、listen/connect endpoint、discovery を設定します。

ホスト固有のendpointや設定を誤ってcommitしないよう、この4つのローカルruntime fileはGitの管理対象外です。`*.example.json5` templateは引き続き追跡します。

独立した各プロジェクト間で、次の設定を一致させます。

| 用途 | NaVILA の設定 | 外部側の設定 | キーの例 |
| --- | --- | --- | --- |
| カメラ JPEG 入力 | `camera.key` | camera の `base_key/device_key` | `camera/front` |
| NaVILA の制御・状態 | 両方の `node_key` | CLI のみ | `navila/command`, `navila/state` |
| Go2 の command・state | `go2.robot_key` | Go2 の `zenoh_key_prefix` | `unitree/go2/command`, `unitree/go2/state` |
| ネットワーク転送 | NaVILA の両 Zenoh file | 外部の両 Zenoh file | peer/router/discovery を一致 |

すべて具体的な Zenoh key である必要があり、wildcard は拒否されます。CLI の設定例は client として `127.0.0.1:7447` の NaVILA ノードへ接続します。CLI が別ホストならこのaddressを書き換えてください。同じホストで複数の peer process を動かす場合、全プロセスを同じ TCP port で listen させないでください。構成に応じて別 listener、multicast discovery、共通 router を使用します。

NaVILA ノードは現在の Go2 `develop` State の `robot_connected`、`robot_state.state`、`accepting_commands` を参照します。旧フィールドの `connected` と `robot` を使う State payload は拒否します。

プログラムが受け取る引数は設定ファイルのパスだけです。model、key、rate、motion 値を個別の CLI 引数で上書きすることはできません。

## 実行

リポジトリルートから推論ノードを起動します。

```console
$ uv run --group zenoh python zenoh/node.py \
    --node-config zenoh/node-config.json5 \
    --zenoh-config zenoh/zenoh-config.json5
```

次に、独立した operator CLI を起動します。

```console
$ uv run --group zenoh python zenoh/cli.py \
    --cli-config zenoh/cli-config.json5 \
    --zenoh-config zenoh/cli-zenoh-config.json5
```

CLI はデフォルトで0.2秒ごとに heartbeat を送ります。NaVILA ノードは heartbeat が fresh になってから `instruction` と `start` を受理しますが、安全操作である `stop` は常に受理します。

操作例です。

```text
navila> ins Go to the elevator
Instruction: Go to the elevator
navila> status
lifecycle=idle connected=True ready=True instruction='Go to the elevator'
navila> start
Start requested; waiting for Go2 ready_stand.
Navigation started.
Running - press Enter to stop
Stopped; Go2 is down.
navila> ins Turn toward the open doorway
navila> start
```

- `ins <text>` は idle または回復可能な error で instruction を変更します。引数なしの `ins` は空文字へ戻します。
- `start` は NaVILA ノードへ stand を要求させ、`ready_stand` を待ってから推論を開始します。
- `status` は最新の NaVILA state を表示します。デフォルトでは state が1秒届かなければ disconnected と表示します。
- running prompt で Enter を押すと stop を送り、down 確認を待って CLI prompt に戻ります。
- `Ctrl+C` で CLI を終了します。idle が確認できない状態では先に `stop` を送り、`command_timeout_sec`（例では15秒）まで待ちます。停止を確認できなくてもその旨を表示し、期限後には終了します。

instruction の初期値は空文字なので、最初は `start` できません。stop/start をまたいで保持されますが、disk には保存されません。

## 周期と画像履歴

カメラ publish、frame sampling、推論、速度 publish は別々の周期です。

| 処理 | 設定例 | 動作 |
| --- | ---: | --- |
| カメラ callback | 外部カメラの周期（通常20–30 Hz） | `image/jpeg` を検証し、推論は行わない |
| 履歴への追加 | 1 Hz | 前回の追加成功から sampling period が経過した callback だけを追加 |
| 推論 | 1 Hz | `running` のときだけ、monotonic な絶対 schedule で実行 |
| 速度 publish | 20 Hz | 現在 action を反復し、action の前後では zero を反復 |
| state publish | 20 Hz | 状態変化がなくても定期 publish |

したがって、別 sampling thread が時計どおりにフレームを抜き出す構成ではありません。カメラ入力が一定ならおおむね等間隔になりますが、厳密な規則は「最後に追加できた時刻から最小間隔を空ける」です。取り逃した interval の catch-up は行わないため、処理待ちの backlog は発生しません。

履歴 deque の長さは `model.config.num_video_frames` から取得し、設定には重複して書きません。そのため8-frame model でも64-frame model でも、checkpoint 自身の長さが使われます。fresh な実画像が1枚あれば start でき、履歴が短い部分は左側を黒画像で padding します。最新の実画像は current observation のままです。idle 中も履歴を更新し、instruction 変更や stop/start をまたいでも保持します。過去の action や model output は prompt に加えません。

推論が周期を超過した場合、過去の slot は飛ばし、推論同士を重ねません。推論中に stop を受けると lifecycle は直ちに `running` から外れ、完了後の結果は破棄されます。停止対象の推論が終わり、Go2 の down が確認されるまで次の start は受理しません。

## lifecycle と安全動作

通常の lifecycle は次のとおりです。

```text
initializing -> idle -> standing -> running -> stopping -> idle
                                                 \-> error
```

start には、fresh な CLI heartbeat、空でない instruction、fresh で有効なカメラ画像、fresh かつ connected で command を受理できる Go2 state、Go2 の `down` 確認が必要です。ノードは `stand` を1回だけ送り、Go2 が `ready_stand` かつ command 受付可能と報告するまで推論も速度 publish も行いません。

running 中は最新の parse 済み action が以前の action を置き換えます。対応出力は `stop`、`move forward`、`turn left`、`turn right` です。距離は25/50/75 cm、旋回は15/30/45 degree に制限します。後退、横移動、不正、曖昧な出力を forward へ fallback することはありません。

operator の Enter、CLI heartbeat 消失、model の `stop`、camera stale、Go2 state の stale/disconnect、parse/inference error、NaVILA の `SIGINT`/`SIGTERM` は、すべて同じ停止経路を通ります。ノードは直ちに `running` を抜け、現在 action を消去し、zero velocity を1回送ります。Go2 state が fresh な場合に限り、reliable な `down` を1回送り、Go2 の `down` と実行中だった推論の両方を待ちます。

Go2 state が0.5秒届かなければ、推測による posture command の送信を止めて `error` へ移ります。`down_timeout_sec` までに down を確認できない場合も、非移動のまま着座未確認を報告します。回復後の `error` から start するには、down 確認を含む全 start 条件が再び有効である必要があります。ノード自身の shutdown も同じ停止経路を使い、安全完了を確認できなければ非ゼロで終了します。

起動後に初めて fresh な Go2 state が届いたとき、ロボットが down でなければ、CLI がなくても `down` を1回要求します。idle 中は Go2 velocity を publish しません。

## 検証

検証は architecture ごとに明確に分けます。x86_64 の開発ホストで確認できるのは、設定と wire model、controller state machine、action parse、frame sampling/padding、schedule 計算、CLI acknowledgement、mock Zenoh による QoS/lifecycle 動作です。これらのテストは fake inference engine を使うため、Jetson の CUDA runtime や実際の NaVILA 推論は検証しません。lock file では platform ごとの選択を保持し、非 aarch64 では通常の PyPI Torch、aarch64 では設定済みの CUDA 13.2 index を選びます。

リポジトリルートから Zenoh 関連テストを実行します。

```console
$ uv run --group zenoh pytest zenoh/tests
```

対象 Jetson では別途、`uv sync --group zenoh`、aarch64 CUDA 版 Torch/Torchvision と Zenoh wheel の import、設定した4bit/8bit/fp16 checkpoint の load、想定 frame 数での推論を確認し、GPU memory と inference latency を計測してください。x86_64 でテストが通っても、これら Jetson 固有項目が通る根拠にはなりません。

実機試験の前に、Jetson 上でロボットなしの録画済み JPEG publisher を使った smoke test を行ってください。その後、Go2 を支持し緊急停止を即座に行える状態で、stand、低速 forward/turn、Enter stop、heartbeat-loss stop、down を確認してください。
