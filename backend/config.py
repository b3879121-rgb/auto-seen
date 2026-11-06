"""
系統設定檔 (System Configuration)
包含 MQTT Broker 連線設定、主題規範 (Topics)、工安告警閾值與視覺模組參數。
"""

import os
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# -------------------------------------------------------------
# MQTT Broker 設定
# -------------------------------------------------------------
# 預設使用公用測試 Broker；有安裝 Mosquitto 可設環境變數 MQTT_BROKER=localhost
MQTT_BROKER = os.getenv("MQTT_BROKER", "broker.emqx.io")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_CLIENT_ID = "SafeFactory_Backend_Server"
MQTT_KEEPALIVE = 60


def _site_id() -> str:
    """每台電腦產生一個固定的隨機站台 ID，避免在公用 Broker 上和別人的主題撞名。"""
    f = ROOT_DIR / ".site_id"
    if f.exists():
        return f.read_text(encoding="utf-8").strip()
    sid = uuid.uuid4().hex[:8]
    f.write_text(sid, encoding="utf-8")
    return sid


SITE_ID = os.getenv("SITE_ID", _site_id())
TOPIC_ROOT = f"safefactory/{SITE_ID}"

# MQTT 主題規範 (Topics)
TOPIC_TELEMETRY_ENV = f"{TOPIC_ROOT}/telemetry/env"          # 環境站：溫濕度、氣體 (QoS 0)
TOPIC_TELEMETRY_MACHINE = f"{TOPIC_ROOT}/telemetry/machine"  # 機台：振動、轉速 (QoS 0)
TOPIC_VISION_ALERTS = f"{TOPIC_ROOT}/vision/alerts"          # YOLO 違規事件 (QoS 1)
TOPIC_CONTROL_ALARM = f"{TOPIC_ROOT}/control/alarm"          # 警報塔燈 / 蜂鳴器指令 (QoS 1)
TOPIC_CONTROL_CAMERA = f"{TOPIC_ROOT}/control/camera"        # 攝影機指令：加強巡檢 (QoS 1)
TOPIC_DEVICE_STATUS = f"{TOPIC_ROOT}/status/devices"         # 裝置上下線 LWT (retained)

# -------------------------------------------------------------
# 工安環境安全閾值 (Safety Thresholds)
# -------------------------------------------------------------
THRESHOLDS = {
    "temperature_max": 40.0,    # °C，超過即觸發高溫告警與加強巡檢
    "temperature_hysteresis": 3.0,  # 降到 (上限 - 3) 以下才解除，避免反覆觸發
    "consecutive_readings": 3,  # 連續幾筆超標才算數 (感測器去雜訊)
    "gas_co2_max": 1200,        # ppm
    "gas_tvoc_max": 500,        # ppb
    "vibration_max": 7.5,       # mm/s
}

# -------------------------------------------------------------
# 視覺模組設定
# -------------------------------------------------------------
PERSON_MODEL = "yolo11n.pt"                               # COCO 預訓練：負責偵測與追蹤「人」
PPE_MODEL = str(ROOT_DIR / "models" / "ppe_yolo11n.pt")   # tools/train_ppe.py 訓練的安全帽/背心模型
CAMERA_INDEX = 0
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEOS_DIR = ROOT_DIR / "videos"          # 放 .mp4 等影片檔即可當作攝影機來源
EVIDENCE_DIR = ROOT_DIR / "evidence"      # 違規存證照片
DEMO_IMAGE_SECONDS = 3.0                  # 示範圖庫模式每張照片停留秒數

# 預設影像來源：demo (資料集照片輪播) / webcam / file:<檔名>
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "demo")

# 一般模式與加強巡檢模式的推論參數
VISION_PROFILES = {
    "normal":   {"conf": 0.45, "infer_every": 2, "window": 12, "min_hits": 8, "cooldown": 30.0},
    "enhanced": {"conf": 0.30, "infer_every": 1, "window": 6,  "min_hits": 4, "cooldown": 10.0},
}
PATROL_SNAPSHOT_SECONDS = 15.0            # 加強巡檢時，每隔幾秒自動拍一張巡檢照
ENHANCED_DEFAULT_DURATION = 120           # 加強巡檢預設持續秒數

# 危險禁區多邊形 (0~1 的比例座標，可在儀表板上重新點選)
FORBIDDEN_ZONE = [
    (0.25, 0.42),
    (0.75, 0.42),
    (0.85, 0.92),
    (0.15, 0.92),
]
