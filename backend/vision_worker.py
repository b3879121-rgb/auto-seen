"""
視覺推論工作線程 (Vision Worker)

流程：影像來源 → YOLO 偵測+追蹤人員 → PPE 模型偵測安全帽/背心 → 依人員位置配對
     → 連續幀確認 (防誤報) → 拍照存證 → 發布 MQTT 違規事件

影像來源：
  demo        Construction-PPE 驗證集照片輪播 (模型沒看過的工地照片)
  webcam      筆電內建鏡頭
  file:<檔名> videos/ 資料夾裡的影片檔，播完自動重播

訂閱 control/camera 指令，收到「加強巡檢」時：降低信心門檻、每幀都推論、
縮短確認時間、定時自動拍巡檢照。
"""

import json
import logging
import random
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

from backend.config import (
    ROOT_DIR,
    PERSON_MODEL,
    PPE_MODEL,
    CAMERA_INDEX,
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    VIDEOS_DIR,
    EVIDENCE_DIR,
    DEMO_IMAGE_SECONDS,
    VIDEO_SOURCE,
    VISION_PROFILES,
    PATROL_SNAPSHOT_SECONDS,
    ENHANCED_DEFAULT_DURATION,
    FORBIDDEN_ZONE,
    TOPIC_VISION_ALERTS,
    TOPIC_CONTROL_CAMERA,
)

logger = logging.getLogger("VisionWorker")

ZONE_FILE = ROOT_DIR / "zone.json"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MIN_PERSON_HEIGHT = 100     # 人框太小時不判定 PPE (看不清楚)
STALE_TRACK_SECONDS = 10.0

EVENT_TEXT = {
    "NO_HELMET": ("WARNING", "偵測到人員未戴安全帽"),
    "NO_VEST": ("WARNING", "偵測到人員未穿反光背心"),
    "ZONE_INTRUSION": ("CRITICAL", "偵測到人員闖入危險禁區"),
    "PATROL_SNAPSHOT": ("INFO", "加強巡檢自動拍照"),
}

GREEN, YELLOW, RED, ORANGE, WHITE = (60, 200, 60), (0, 200, 255), (0, 0, 255), (0, 140, 255), (255, 255, 255)


def letterbox(frame: np.ndarray, w: int = VIDEO_WIDTH, h: int = VIDEO_HEIGHT) -> np.ndarray:
    """等比例縮放並補黑邊，避免畫面變形。"""
    fh, fw = frame.shape[:2]
    s = min(w / fw, h / fh)
    nw, nh = int(fw * s), int(fh * s)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(frame, (nw, nh))
    return out


def demo_image_dir() -> Path:
    return Path(SETTINGS["datasets_dir"]) / "construction-ppe" / "images" / "val"


class FrameSource:
    """統一封裝三種影像來源，read() 一律回傳 640x480 BGR 影像或 None。"""

    def __init__(self, spec: str):
        self.spec = spec
        self.cap = None
        self.images = []
        self.img_idx = 0
        self.img_since = 0.0
        self.current_img = None
        self.frame_interval = 0.0
        self.last_read = 0.0

        if spec == "demo":
            d = demo_image_dir()
            self.images = sorted(p for p in d.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) if d.exists() else []
            random.shuffle(self.images)
            if not self.images:
                raise RuntimeError(f"找不到示範圖庫 ({d})，請先執行 tools/train_ppe.py 下載資料集")
        elif spec == "webcam":
            self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                raise RuntimeError("無法開啟筆電鏡頭")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
        elif spec.startswith("file:"):
            path = VIDEOS_DIR / spec[5:]
            self.cap = cv2.VideoCapture(str(path))
            if not self.cap.isOpened():
                raise RuntimeError(f"無法開啟影片 {path}")
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
            self.frame_interval = 1.0 / min(max(fps, 1), 60)
        else:
            raise RuntimeError(f"未知的影像來源: {spec}")

    @property
    def label(self) -> str:
        if self.spec == "demo":
            return f"DEMO {self.img_idx + 1}/{len(self.images)}"
        return self.spec.upper()

    def read(self):
        if self.images:
            now = time.time()
            if self.current_img is None or now - self.img_since >= DEMO_IMAGE_SECONDS:
                if self.current_img is not None:
                    self.img_idx = (self.img_idx + 1) % len(self.images)
                img = cv2.imread(str(self.images[self.img_idx]))
                self.current_img = letterbox(img) if img is not None else None
                self.img_since = now
            return None if self.current_img is None else self.current_img.copy()

        # 影片檔依原始 FPS 播放，避免 GPU 太快導致快轉
        if self.frame_interval:
            wait = self.frame_interval - (time.time() - self.last_read)
            if wait > 0:
                time.sleep(wait)
            self.last_read = time.time()
        ok, frame = self.cap.read()
        if not ok and self.spec.startswith("file:"):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return letterbox(frame) if ok else None

    def release(self):
        if self.cap is not None:
            self.cap.release()


class VisionWorker:
    def __init__(self, mqtt_manager):
        self.mqtt = mqtt_manager
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.latest_frame_bytes = None

        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

        logger.info(f"載入人員模型 {PERSON_MODEL} (裝置: {self.device_name})")
        self.person_model = YOLO(PERSON_MODEL)
        self.ppe_model = None
        if Path(PPE_MODEL).exists():
            self.ppe_model = YOLO(PPE_MODEL)
            logger.info(f"✅ 已載入 PPE 模型: {self.ppe_model.names}")
        else:
            logger.warning("⚠️ 尚未訓練 PPE 模型 (models/ppe_yolo11n.pt)，只啟用禁區偵測")

        # 影像來源
        self.source_spec = VIDEO_SOURCE
        self.requested_source = None
        self.source_error = ""

        # 推論模式 (normal / enhanced)
        self.mode = "normal"
        self.enhanced_until = 0.0
        self.enhanced_reason = ""
        self.last_patrol = 0.0

        # 禁區 (比例座標)
        self.zone_norm = self._load_zone()

        # 防誤報狀態
        self.history = {}       # (track_id, event) -> deque[bool]
        self.last_alert = {}    # (track_id, event) -> time
        self.last_seen = {}     # track_id -> time

        # 統計
        self.fps = 0.0
        self.infer_fps = 0.0
        self.current_person_count = 0
        self.is_danger = False
        self.counts = {"NO_HELMET": 0, "NO_VEST": 0, "ZONE_INTRUSION": 0}

        # 事件紀錄
        EVIDENCE_DIR.mkdir(exist_ok=True)
        self.events_file = EVIDENCE_DIR / "events.jsonl"
        self.events = deque(self._load_events(), maxlen=200)

        mqtt_manager.register_callback(self.on_mqtt_message)

    # ---------------------------------------------------------
    # 設定：禁區、來源、MQTT 指令
    # ---------------------------------------------------------
    def _load_zone(self):
        try:
            pts = json.loads(ZONE_FILE.read_text(encoding="utf-8"))
            if len(pts) >= 3:
                return [tuple(p) for p in pts]
        except Exception:
            pass
        return list(FORBIDDEN_ZONE)

    def set_zone(self, points):
        pts = [(min(max(float(x), 0.0), 1.0), min(max(float(y), 0.0), 1.0)) for x, y in points]
        if len(pts) < 3:
            raise ValueError("禁區至少需要 3 個點")
        self.zone_norm = pts
        ZONE_FILE.write_text(json.dumps(pts), encoding="utf-8")

    def zone_px(self):
        return np.array([(int(x * VIDEO_WIDTH), int(y * VIDEO_HEIGHT)) for x, y in self.zone_norm], np.int32)

    @staticmethod
    def list_sources():
        sources = []
        if demo_image_dir().exists():
            sources.append({"id": "demo", "label": "示範圖庫 (工地照片)"})
        sources.append({"id": "webcam", "label": "筆電鏡頭"})
        if VIDEOS_DIR.exists():
            for p in sorted(VIDEOS_DIR.iterdir()):
                if p.suffix.lower() in VIDEO_EXTS:
                    sources.append({"id": f"file:{p.name}", "label": f"影片：{p.name}"})
        return sources

    def request_source(self, spec: str):
        self.requested_source = spec

    def on_mqtt_message(self, topic: str, data: dict):
        """IoT → AI：收到 control/camera 指令切換巡檢模式。"""
        if topic != TOPIC_CONTROL_CAMERA or data.get("command") != "ENHANCED_INSPECTION":
            return
        if data.get("enable"):
            duration = float(data.get("duration_s", ENHANCED_DEFAULT_DURATION))
            self.mode = "enhanced"
            self.enhanced_until = time.time() + duration
            self.enhanced_reason = data.get("reason", "")
            self.last_patrol = 0.0
            logger.warning(f"🔎 進入加強巡檢模式 {duration:.0f}s：{self.enhanced_reason}")
        else:
            self._exit_enhanced(data.get("reason", "指令解除"))

    def _exit_enhanced(self, reason: str):
        if self.mode == "enhanced":
            logger.info(f"✅ 解除加強巡檢：{reason}")
        self.mode = "normal"
        self.enhanced_until = 0.0
        self.enhanced_reason = ""

    # ---------------------------------------------------------
    # 偵測與判斷
    # ---------------------------------------------------------
    def _detect(self, frame, profile):
        """回傳人員清單：[{tid, box, conf, helmet, vest, in_zone}]"""
        res = self.person_model.track(
            frame, persist=True, classes=[0], conf=profile["conf"],
            tracker="bytetrack.yaml", device=self.device, verbose=False,
        )[0]

        ppe = []
        if self.ppe_model is not None:
            r = self.ppe_model(frame, conf=profile["conf"], device=self.device, verbose=False)[0]
            for b in r.boxes:
                name = self.ppe_model.names[int(b.cls[0])]
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                ppe.append((name, (x1 + x2) / 2, (y1 + y2) / 2, (x1, y1, x2, y2)))

        zone = self.zone_px()
        people = []
        for i, b in enumerate(res.boxes):
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            w, h = x2 - x1, y2 - y1
            tid = int(b.id[0]) if b.id is not None else -(i + 1)  # 尚未確認的軌跡給負數暫時 ID

            def match(names, rx1, ry1, rx2, ry2):
                return any(n in names and rx1 <= cx <= rx2 and ry1 <= cy <= ry2 for n, cx, cy, _ in ppe)

            helmet = vest = "unknown"
            if self.ppe_model is not None:
                # 頭部區域：人框上方 35% (往外擴一點)；軀幹區域：15%~75%
                head = (x1 - 0.1 * w, y1 - 0.15 * h, x2 + 0.1 * w, y1 + 0.35 * h)
                torso = (x1, y1 + 0.15 * h, x2, y1 + 0.75 * h)
                big = h >= MIN_PERSON_HEIGHT
                if match({"helmet"}, *head):
                    helmet = "ok"
                elif match({"no_helmet"}, *head) or big:
                    helmet = "missing"
                if match({"vest"}, *torso):
                    vest = "ok"
                elif match({"no_vest"}, *torso) or big:
                    vest = "missing"

            foot = ((x1 + x2) // 2, y2)  # 腳底接觸點判斷是否進入禁區
            in_zone = cv2.pointPolygonTest(zone, foot, False) >= 0
            people.append({"tid": tid, "box": (x1, y1, x2, y2), "conf": float(b.conf[0]),
                           "helmet": helmet, "vest": vest, "in_zone": in_zone, "foot": foot})
        return people, ppe

    def _confirm(self, people, profile, now):
        """連續幀確認：最近 window 幀中至少 min_hits 幀違規才成立，並套用冷卻時間。"""
        confirmed = []
        for p in people:
            tid = p["tid"]
            if tid < 0:
                continue
            self.last_seen[tid] = now
            flags = {
                "NO_HELMET": p["helmet"] == "missing",
                "NO_VEST": p["vest"] == "missing",
                "ZONE_INTRUSION": p["in_zone"],
            }
            for event, bad in flags.items():
                key = (tid, event)
                hist = self.history.get(key)
                if hist is None or hist.maxlen != profile["window"]:
                    hist = deque(hist or [], maxlen=profile["window"])
                    self.history[key] = hist
                hist.append(bad)
                if bad and sum(hist) >= profile["min_hits"]:
                    if now - self.last_alert.get(key, 0) >= profile["cooldown"]:
                        self.last_alert[key] = now
                        confirmed.append((event, p))

        # 清掉離開畫面的軌跡
        for tid in [t for t, ts in self.last_seen.items() if now - ts > STALE_TRACK_SECONDS]:
            self.last_seen.pop(tid, None)
            for k in [k for k in self.history if k[0] == tid]:
                self.history.pop(k, None)
        return confirmed

    # ---------------------------------------------------------
    # 繪圖
    # ---------------------------------------------------------
    def _draw(self, frame, people, ppe, source_label, now):
        out = frame.copy()
        zone = self.zone_px()
        danger = any(p["in_zone"] for p in people)

        overlay = out.copy()
        cv2.fillPoly(overlay, [zone], RED if danger else (30, 30, 200))
        cv2.addWeighted(overlay, 0.40 if danger else 0.22, out, 0.60 if danger else 0.78, 0, out)
        cv2.polylines(out, [zone], True, RED if danger else (60, 60, 230), 2)
        zx, zy = zone[:, 0].min(), zone[:, 1].min()
        cv2.putText(out, "DANGER ZONE", (int(zx) + 6, int(zy) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2)

        for name, _, _, (x1, y1, x2, y2) in ppe:
            if name in ("helmet", "vest", "no_helmet", "no_vest"):
                c = GREEN if name in ("helmet", "vest") else RED
                cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), c, 1)

        for p in people:
            x1, y1, x2, y2 = p["box"]
            bad = p["in_zone"] or p["helmet"] == "missing" or p["vest"] == "missing"
            color = RED if p["in_zone"] else (YELLOW if bad else GREEN)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.circle(out, p["foot"], 4, color, -1)
            tag = {"ok": "OK", "missing": "NO", "unknown": "?"}
            label = f"ID{p['tid']}" if p["tid"] >= 0 else "new"
            if self.ppe_model is not None:
                label += f" H:{tag[p['helmet']]} V:{tag[p['vest']]}"
            if p["in_zone"]:
                label += " ZONE!"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (x1, max(y1 - th - 8, 0)), (x1 + tw + 6, max(y1, th + 8)), color, -1)
            cv2.putText(out, label, (x1 + 3, max(y1 - 5, th + 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # 加強巡檢：橘色外框 + 倒數
        if self.mode == "enhanced":
            cv2.rectangle(out, (0, 0), (VIDEO_WIDTH - 1, VIDEO_HEIGHT - 1), ORANGE, 6)
            remain = max(0, int(self.enhanced_until - now))
            cv2.rectangle(out, (0, 0), (VIDEO_WIDTH, 30), ORANGE, -1)
            cv2.putText(out, f"ENHANCED INSPECTION (HIGH TEMP)  {remain}s", (12, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        elif danger:
            cv2.rectangle(out, (0, 0), (VIDEO_WIDTH, 30), RED, -1)
            cv2.putText(out, "!!! WORKER IN DANGER ZONE !!!", (150, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

        hud = f"{source_label} | FPS {self.fps:.0f} | AI {self.infer_fps:.0f}Hz | {self.mode.upper()} | {self.device_name}"
        cv2.rectangle(out, (0, VIDEO_HEIGHT - 24), (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0), -1)
        cv2.putText(out, hud, (8, VIDEO_HEIGHT - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        return out, danger

    # ---------------------------------------------------------
    # 存證與事件
    # ---------------------------------------------------------
    def _load_events(self):
        if not self.events_file.exists():
            return []
        lines = self.events_file.read_text(encoding="utf-8").splitlines()[-200:]
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
        return out

    def _save_evidence(self, annotated, event: str, person=None):
        now = datetime.now()
        day = now.strftime("%Y%m%d")
        folder = EVIDENCE_DIR / day
        folder.mkdir(parents=True, exist_ok=True)
        tid = person["tid"] if person else None
        name = f"{now.strftime('%H%M%S_%f')[:-3]}_{event}" + (f"_id{tid}" if tid is not None else "") + ".jpg"

        img = annotated.copy()
        if person:
            x1, y1, x2, y2 = person["box"]
            cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 0, 255), 3)
        stamp = now.strftime("%Y-%m-%d %H:%M:%S") + f"  CAM_01  {event}"
        cv2.putText(img, stamp, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(img, stamp, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
        cv2.imwrite(str(folder / name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])

        severity, message = EVENT_TEXT[event]
        record = {
            "id": f"{day}-{name[:-4]}",
            "event": event,
            "severity": severity,
            "camera_id": "CAM_01",
            "zone": "ZONE_A",
            "timestamp": time.time(),
            "track_id": tid,
            "confidence": round(person["conf"], 2) if person else None,
            "mode": self.mode,
            "snapshot_url": f"/evidence/{day}/{name}",
            "message": message + (f" (ID {tid})" if tid is not None else ""),
        }
        self.events.append(record)
        with self.events_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.mqtt.publish(TOPIC_VISION_ALERTS, record, qos=1)
        return record

    # ---------------------------------------------------------
    # 主迴圈
    # ---------------------------------------------------------
    def _open_source(self, spec):
        try:
            src = FrameSource(spec)
            self.source_spec = spec
            self.source_error = ""
            # 換來源時重設追蹤器，避免舊 ID 延續
            if getattr(self.person_model, "predictor", None) is not None:
                for t in getattr(self.person_model.predictor, "trackers", []) or []:
                    t.reset()
            self.history.clear()
            logger.info(f"🎥 影像來源：{spec}")
            return src
        except Exception as e:
            self.source_error = str(e)
            logger.error(f"影像來源 {spec} 開啟失敗：{e}")
            return None

    def _no_signal(self):
        img = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), np.uint8)
        cv2.putText(img, "NO SIGNAL", (215, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 80, 255), 3)
        cv2.putText(img, self.source_error[:70].encode("ascii", "ignore").decode(), (20, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        return img

    def run_inference(self):
        src = self._open_source(self.source_spec)
        if src is None and self.source_spec != "webcam":
            src = self._open_source("webcam")

        frame_no = 0
        people, ppe = [], []
        win_start, win_frames, win_infers = time.time(), 0, 0

        while self.running:
            if self.requested_source:
                if src:
                    src.release()
                src = self._open_source(self.requested_source)
                self.requested_source = None

            now = time.time()
            # 每秒統計一次畫面幀率與 AI 推論頻率
            if now - win_start >= 1.0:
                self.fps = win_frames / (now - win_start)
                self.infer_fps = win_infers / (now - win_start)
                win_start, win_frames, win_infers = now, 0, 0

            if self.mode == "enhanced" and now > self.enhanced_until:
                self._exit_enhanced("巡檢時間到")
            profile = VISION_PROFILES[self.mode]

            frame = src.read() if src else None
            if frame is None:
                self._publish_frame(self._no_signal())
                time.sleep(0.2)
                continue

            frame_no += 1
            confirmed = []
            if frame_no % profile["infer_every"] == 0:
                try:
                    people, ppe = self._detect(frame, profile)
                    confirmed = self._confirm(people, profile, now)
                except Exception as e:
                    logger.exception(f"推論錯誤：{e}")
                    people, ppe = [], []
                win_infers += 1
            win_frames += 1

            annotated, danger = self._draw(frame, people, ppe, src.label, now)
            self.current_person_count = len(people)
            self.is_danger = danger

            for event, person in confirmed:
                self.counts[event] += 1
                rec = self._save_evidence(annotated, event, person)
                logger.warning(f"🚨 {rec['message']} → 已存證並發布 MQTT")

            if self.mode == "enhanced" and now - self.last_patrol >= PATROL_SNAPSHOT_SECONDS:
                self.last_patrol = now
                self._save_evidence(annotated, "PATROL_SNAPSHOT")

            self._publish_frame(annotated)
            time.sleep(0.005)

        if src:
            src.release()
        logger.info("🛑 視覺推論迴圈已關閉")

    def _publish_frame(self, img):
        ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ok:
            with self.lock:
                self.latest_frame_bytes = jpg.tobytes()

    # ---------------------------------------------------------
    def status(self):
        return {
            "fps": round(self.fps, 1),
            "infer_fps": round(self.infer_fps, 1),
            "device": self.device_name,
            "persons": self.current_person_count,
            "is_danger": self.is_danger,
            "mode": self.mode,
            "enhanced_remaining": max(0, int(self.enhanced_until - time.time())) if self.mode == "enhanced" else 0,
            "enhanced_reason": self.enhanced_reason,
            "source": self.source_spec,
            "source_error": self.source_error,
            "ppe_model_loaded": self.ppe_model is not None,
            "counts": self.counts,
            "zone": self.zone_norm,
        }

    def get_frame(self) -> bytes:
        with self.lock:
            return self.latest_frame_bytes

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.run_inference, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)
