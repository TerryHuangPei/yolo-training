# YOLO Training Pipeline 指令手冊

本文件整理本專案目前提供的 CLI、Make 與 Docker 指令。除非另有說明，請先在專案根目錄執行。

## 快速開始：原生 macOS

本專案在 Apple Silicon Mac 上會自動優先使用 MPS，沒有可用 MPS 時改用 CPU。

```sh
# 初次安裝（需要 Python 3.11；會建立 .venv）
brew install python@3.11
make mac-setup

# 確認 Python、PyTorch 與 MPS 環境
make mac-doctor

# 顯示所有 CLI 指令與參數
.venv/bin/yolo-pipeline --help
```

若你已經有 `.venv`，且從舊版專案更新而來，請重新安裝本專案相依套件，讓 `mss`（螢幕擷取）一併安裝：

```sh
.venv/bin/pip install -e .
```

## 快速開始：原生 Windows

在 PowerShell 執行。需要 Python 3.11；首次安裝會建立 `.venv`、安裝專案相依套件並列出 CUDA/CPU 狀態。

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup-windows.ps1
.venv\Scripts\python.exe scripts\windows_doctor.py
.venv\Scripts\yolo-pipeline.exe --help
```

若使用 NVIDIA GPU，`windows-doctor` 顯示 `cuda_available=True` 時會自動選擇 CUDA。否則使用 CPU；可在訓練時明確指定 `--device 0` 或 `--device cpu`。`mps` 僅適用 macOS，不可用於 Windows。

```powershell
.venv\Scripts\yolo-pipeline.exe pipeline `
  --dataset input\dataset.zip `
  --name helmet-v1 `
  --epochs 100 `
  --imgsz 640
```

## 常用：即時螢幕辨識 `predict-screen`

以訓練完成的 `best.pt` 即時判別電腦螢幕。程式會開啟帶有偵測框的預覽視窗；按 `q` 或 `Esc` 即可停止。

```sh
.venv/bin/yolo-pipeline predict-screen \
  --model artifacts/jobs/JOB_ID/model/best.pt
```

常用調整範例：

```sh
# 第二台實體螢幕、提高最低信心分數、指定 MPS（macOS）
.venv/bin/yolo-pipeline predict-screen \
  --model artifacts/jobs/JOB_ID/model/best.pt \
  --monitor 2 \
  --conf 0.40 \
  --imgsz 640 \
 --device mps
```

預覽視窗只用來顯示辨識結果；不必點選它。維持原本要操作的應用程式視窗為作用中視窗，無論滑鼠在哪裡，按下全域熱鍵 `` ` ``，程式就會計算滑鼠當下位置到畫面中每一個 `head` 框中心的像素距離，並立即附加一筆 JSON 記錄到：

```text
artifacts/jobs/JOB_ID/screen-clicks.jsonl
```

預覽畫面會以紅色十字顯示目前系統游標及其座標。這是額外繪製的 overlay，因為螢幕擷取本身通常不會包含系統游標；它不會影響原本應用程式中的游標。

每筆記錄含按鍵觸發時間（`triggered_at`）、滑鼠與中心的畫面座標（`x`、`y`）、對應的全螢幕座標（`screen_x`、`screen_y`）、目標框、信心分數、影格編號和 `distance_pixels`。一次熱鍵觸發會對畫面中每個 `head` 各寫一筆；即使游標在框外也會記錄。若該畫面沒有 `head`，才不會寫入資料。若你的類別名稱不是 `head`、想換觸發按鍵或自訂紀錄檔位置：

```sh
.venv/bin/yolo-pipeline predict-screen \
  --model artifacts/jobs/JOB_ID/model/best.pt \
  --target-class helmet \
  --hotkey g \
  --output artifacts/jobs/JOB_ID/helmet-clicks.jsonl
```

要在紀錄後直接套用 `move_mouse()`，加入 `--move-to-head`。程式會將記錄中的
`mouse_position.screen_x/y` 與 `head_center.screen_x/y` 直接傳入 `move_mouse()`；若同時
偵測到多個 head，會移向距離游標最近的一個。

```sh
.venv/bin/yolo-pipeline predict-screen \
  --model artifacts/jobs/JOB_ID/model/best.pt \
  --move-to-head \
  --move-fov 90 \
  --move-sensitivity 1.5 \
  --move-duration 0.2 \
  --move-steps 20
```

參數說明：

| 參數 | 預設值 | 說明 |
| --- | --- | --- |
| `--model` | 必填 | `.pt` 模型路徑，通常是工作產物中的 `model/best.pt`。 |
| `--monitor` | `1` | 實體螢幕編號，第一台是 `1`、第二台是 `2`。 |
| `--conf` | `0.25` | 最低偵測信心分數；提高可減少誤判，但可能漏判。 |
| `--imgsz` | `640` | 模型推論影像尺寸；降低（如 `416`）通常較快，但精度可能下降。 |
| `--device` | 自動選擇 | macOS 可指定 `mps`/`cpu`；Windows 可指定 CUDA 的 `0` 或 `cpu`。 |
| `--target-class` | `head` | 熱鍵觸發時要量測的模型類別名稱。 |
| `--hotkey` | `` ` `` | 一個字元的全域觸發按鍵。 |
| `--output` | 模型所在工作下的 `screen-clicks.jsonl` | 熱鍵觸發時的座標與距離 JSONL 輸出位置。 |
| `--move-to-head` | `False` | 將滑鼠移向最近的目標 head 中心。 |
| `--move-fov` | `90` | 傳給 `move_mouse()` 的目前 FOV。 |
| `--move-sensitivity` | `1` | 傳給 `move_mouse()` 的靈敏度。 |
| `--move-duration` | `0.2` | 移動所需約略秒數。 |
| `--move-steps` | `20` | 平滑移動的分段數。 |

> macOS 第一次使用時，請在「系統設定 → 隱私權與安全性 → 螢幕與系統音訊錄製」和「輔助使用」授權啟動指令的 Terminal 或 IDE。前者讓程式讀取螢幕，後者讓全域熱鍵能在原本應用程式作用中時運作。授權後請重新啟動指令。

> Windows 使用 `predict-screen` 時，請以與目標程式相同的權限層級執行 PowerShell/IDE；一般權限的 Python 無法控制以系統管理員身分執行的程式，也無法操作鎖定畫面或 UAC 安全桌面。程式已啟用 per-monitor DPI awareness，使擷取畫面與游標座標在縮放、多螢幕環境中對齊。Windows 指令請將 `.venv/bin/yolo-pipeline` 改為 `.venv\Scripts\yolo-pipeline.exe`，路徑分隔符號改為 `\`。此功能必須原生執行，Docker 無法讀取宿主機桌面。

## 常用：完整訓練流程 `pipeline`

`pipeline` 會依序驗證資料集、建立 train/val 切分（若尚未切分）、訓練、驗證、提升 `best.pt`；也可在完成後對指定影片進行推論。

```sh
.venv/bin/yolo-pipeline pipeline \
  --dataset input/dataset.zip \
  --name helmet-v1 \
  --epochs 100 \
  --imgsz 640
```

也可使用 Make 的簡寫（固定使用 `input/dataset.zip` 與名稱 `example`）：

```sh
make mac-pipeline
```

訓練完成後，終端機會輸出 `JOB_ID`。模型通常位於：

```text
artifacts/jobs/JOB_ID/model/best.pt
```

完整範例（訓練後直接判別預錄影片，並要求最低 mAP）：

```sh
.venv/bin/yolo-pipeline pipeline \
  --dataset input/dataset.zip \
  --name helmet-v2 \
  --model yolo26n.pt \
  --epochs 150 \
  --imgsz 640 \
  --device mps \
  --batch 8 \
  --min-map50 0.70 \
  --min-map50-95 0.45 \
  --video input/test.mp4
```

`pipeline` 參數：

| 參數 | 預設值 | 說明 |
| --- | --- | --- |
| `--dataset` | 必填 | 資料集 ZIP 或資料集目錄。 |
| `--name` | 必填 | 本次工作名稱，會用於 `JOB_ID`。 |
| `--model` | `yolo26n.pt` | 訓練的基底模型。 |
| `--epochs` | `100` | 訓練輪數。 |
| `--imgsz` | `640` | 訓練與驗證圖片尺寸。 |
| `--device` | 自動選擇 | macOS 為 `mps`/`cpu`；Windows 為 `0`（CUDA GPU）/`cpu`。 |
| `--batch` | 自動 | 每批圖片數。MPS／CPU 預設為 `8`。 |
| `--min-map50` | 不限制 | 低於此 mAP@0.5 時，模型不會被提升為成果。 |
| `--min-map50-95` | 不限制 | 低於此 mAP@0.5:0.95 時，模型不會被提升為成果。 |
| `--video` | 無 | 訓練完成後，以 `best.pt` 對此影片產生標註影片與 JSONL。 |

## 單獨訓練 `train`

只執行資料集準備、訓練與驗證流程：

```sh
.venv/bin/yolo-pipeline train \
  --dataset input/dataset.zip \
  --name experiment-01 \
  --epochs 100 \
  --imgsz 640
```

從中斷的訓練續跑：

```sh
.venv/bin/yolo-pipeline train \
  --resume artifacts/jobs/JOB_ID/model/last.pt
```

如資料集位置無法從舊工作推得，可同時提供 `--dataset`。

## 使用既有模型判別預錄影片 `predict-video`

```sh
.venv/bin/yolo-pipeline predict-video \
  --model artifacts/jobs/JOB_ID/model/best.pt \
  --source input/test.mp4 \
  --conf 0.25 \
  --imgsz 640
```

若模型檔名是 `best.pt`，輸出會寫入同一份工作下的 `inference/`，包括：

```text
artifacts/jobs/JOB_ID/inference/annotated-video.mp4
artifacts/jobs/JOB_ID/inference/detections.jsonl
artifacts/jobs/JOB_ID/inference/summary.json
```

## 資料集檢查與工作查詢

```sh
# 只驗證資料集格式與標籤；會建立一份 dataset-validation 工作
.venv/bin/yolo-pipeline dataset validate --dataset input/dataset.zip

# 檢視既有工作狀態、設定與錯誤資訊
.venv/bin/yolo-pipeline job show --job-id JOB_ID
```

## 開發檢查與格式化

```sh
# 原生 macOS 環境
make mac-test
make mac-lint
make mac-format

# 也可以直接使用虛擬環境
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/ruff format .
```

Windows PowerShell 對應指令：

```powershell
.venv\Scripts\pytest.exe
.venv\Scripts\ruff.exe check .
.venv\Scripts\ruff.exe format .
```

## 在其他 Python 程式重用滑鼠移動函式

滑鼠計算與控制封裝在 `app.mouse`，可以直接 import：

```python
from app.mouse import move_mouse

actual_dx, actual_dy = move_mouse(
    point_a=(960, 540),
    point_b=(1200, 600),
    fov=90.0,
    sensitivity=1.5,
    duration=0.2,
    steps=20,
    smooth=True,
)
```

內部分成 `calculate_mouse_delta()`（座標、FOV、靈敏度）、`generate_mouse_path()`（linear 或 ease-in-out 路徑）與 `move_mouse()`（透過可替換的 backend 執行）。預設 backend 為 `pynput`；在測試或改用其他平台 API 時，傳入具有 `move_relative(dx, dy)` 的物件即可。FOV 的正規化公式集中在 `calculate_mouse_delta()` 的 `fov_scale`，要校正特定遊戲時只需替換該段公式。

## Docker 指令

Docker 會將本機 `input/`、`workspace/`、`artifacts/` 分別掛載至容器的 `/workspace/input`、`/workspace/jobs`、`/workspace/artifacts`。

```sh
# CPU 映像檔建置與檢查
make build-cpu
make test
make lint

# CPU 執行完整訓練流程
docker compose -f compose.yaml -f compose.cpu.yaml run --rm yolo-pipeline \
  pipeline \
  --dataset /workspace/input/dataset.zip \
  --name helmet-v1 \
  --epochs 100 \
  --imgsz 640

# NVIDIA GPU 執行完整流程（需已安裝 NVIDIA Container Toolkit）
docker compose run --rm yolo-pipeline \
  pipeline \
  --dataset /workspace/input/dataset.zip \
  --name helmet-v1

# 在容器內判別預錄影片
docker compose -f compose.yaml -f compose.cpu.yaml run --rm yolo-pipeline \
  predict-video \
  --model /workspace/artifacts/jobs/JOB_ID/model/best.pt \
  --source /workspace/input/test.mp4

# 檢視工作狀態
docker compose -f compose.yaml -f compose.cpu.yaml run --rm yolo-pipeline \
  job show --job-id JOB_ID

# 進入容器 shell
make shell
```

> 不要在 Docker 中使用 `predict-screen`；容器無法直接讀取 macOS 或 Windows 的宿主機桌面。

## 工作資料夾與清理

每次訓練或驗證都會建立唯一的 `JOB_ID`。重要結果位於：

```text
artifacts/jobs/JOB_ID/model/best.pt        # 已提升的最佳模型
artifacts/jobs/JOB_ID/model/last.pt        # 最後一次訓練的模型
artifacts/jobs/JOB_ID/reports/             # 驗證指標、訓練摘要
artifacts/jobs/JOB_ID/logs/pipeline.log    # 執行日誌
artifacts/jobs/JOB_ID/manifest.json        # 工作狀態與中繼資料
workspace/JOB_ID/                          # 工作暫存資料與訓練輸出
```

若確定不再需要某一筆工作，可在 Docker 模式清理指定資料夾（必須明確提供 `JOB_ID`）：

```sh
make clean-workspace JOB_ID=JOB_ID
make clean-artifacts JOB_ID=JOB_ID
```

`clean-artifacts` 會刪除該工作的模型、報告與推論結果，無法自動復原，請先確認目標工作編號。
