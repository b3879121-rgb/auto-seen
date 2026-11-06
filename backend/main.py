"""
FastAPI 主伺服器 (SafeFactory AIoT Backend)
提供即時 MJPEG 影像串流、WebSocket 推播、REST 控制 API，
並把 MQTT 訊息分派給規則引擎、虛擬塔燈與攝影機。
"""

import asyncio
import json
import logging
import os
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from typing import List, Set, Tuple
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import (
    TOPIC_ROOT,
    TOPIC_CONTROL_ALARM,
    TOPIC_CONTROL_CAMERA,
    THRESHOLDS,
    EVIDENCE_DIR,
    ENHANCED_DEFAULT_DURATION,
)
from backend.mqtt_manager import MQTTManager
from backend.sensor_simulator import SensorSimulator
from backend.vision_worker import VisionWorker
from backend.rule_engine import RuleEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SafeFactoryServer")

app = FastAPI(title="SafeFactory AIoT System", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 核心實例：所有元件都透過 MQTT 溝通
mqtt_mgr = MQTTManager()
sensor_sim = SensorSimulator(mqtt_mgr)             # 虛擬環境站 + 虛擬警報塔燈
vision_worker = VisionWorker(mqtt_mgr)              # 攝影機 + YOLO (自行訂閱 control/camera)
rules = RuleEngine(mqtt_mgr, sensor_sim)            # 連動規則
mqtt_mgr.register_callback(sensor_sim.on_mqtt_message)
mqtt_mgr.register_callback(rules.on_mqtt_message)

active_websockets: Set[WebSocket] = set()
loop: asyncio.AbstractEventLoop = None


def channel_of(topic: str) -> str:
    """把完整主題轉成短名稱 (例如 telemetry/env) 給前端判斷。"""
    return topic[len(TOPIC_ROOT) + 1:] if topic.startswith(TOPIC_ROOT + "/") else topic


def broadcast_mqtt_to_ws(topic: str, data: dict):
    if loop and loop.is_running() and active_websockets:
        message = json.dumps({"type": "mqtt_message", "topic": topic, "channel": channel_of(topic), "data": data})
        asyncio.run_coroutine_threadsafe(_send_ws_broadcast(message), loop)


async def _send_ws_broadcast(message: str):
    for ws in list(active_websockets):
        try:
            await ws.send_text(message)
        except Exception:
            active_websockets.discard(ws)


mqtt_mgr.register_callback(broadcast_mqtt_to_ws)


@app.on_event("startup")
async def startup_event():
    global loop
    loop = asyncio.get_running_loop()
    logger.info(f"🚀 啟動 SafeFactory 後端服務 (MQTT 主題根目錄: {TOPIC_ROOT}/#)")
    mqtt_mgr.start()
    sensor_sim.start()
    vision_worker.start()


@app.on_event("shutdown")
def shutdown_event():
    sensor_sim.stop()
    vision_worker.stop()
    mqtt_mgr.stop()


# -------------------------------------------------------------
# REST API
# -------------------------------------------------------------
class EnableRequest(BaseModel):
    enable: bool


class SourceRequest(BaseModel):
    source: str


class ZoneRequest(BaseModel):
    points: List[Tuple[float, float]]


@app.get("/api/status")
async def get_system_status():
    return {
        "mqtt_connected": mqtt_mgr.is_connected,
        "broker": mqtt_mgr.broker,
        "topic_root": TOPIC_ROOT,
        "tower_light": sensor_sim.tower_light,
        "buzzer": sensor_sim.buzzer,
        "alarm_reason": sensor_sim.last_alarm_reason,
        "anomaly_mode": sensor_sim.anomaly_mode,
        "temp_high": rules.temp_high,
        "co2_high": rules.co2_high,
        "vision": vision_worker.status(),
        "thresholds": THRESHOLDS,
    }


@app.post("/api/control/alarm_reset")
async def reset_alarm():
    """手動復歸：關閉異常模擬、塔燈回綠燈、解除加強巡檢"""
    sensor_sim.anomaly_mode = False
    rules.reset()
    mqtt_mgr.publish(TOPIC_CONTROL_ALARM, {"command": "RESET_ALARM", "timestamp": time.time()}, qos=1)
    mqtt_mgr.publish(TOPIC_CONTROL_CAMERA, {"command": "ENHANCED_INSPECTION", "enable": False,
                                            "reason": "手動復歸", "timestamp": time.time()}, qos=1)
    return {"status": "success", "message": "警報已手動復歸"}


@app.post("/api/control/buzzer_mute")
async def mute_buzzer():
    mqtt_mgr.publish(TOPIC_CONTROL_ALARM, {"command": "MUTE_BUZZER", "timestamp": time.time()}, qos=1)
    return {"status": "success", "message": "蜂鳴器已消音"}


@app.post("/api/control/anomaly")
async def toggle_anomaly(req: EnableRequest):
    """注入感測異常 (高溫、CO2、振動)，用來展示 IoT → AI 連動"""
    sensor_sim.trigger_anomaly(req.enable)
    return {"status": "success", "anomaly_mode": req.enable}


@app.post("/api/control/inspection")
async def manual_inspection(req: EnableRequest):
    """手動發布加強巡檢指令 (和規則引擎發的是同一個 MQTT 指令)"""
    mqtt_mgr.publish(TOPIC_CONTROL_CAMERA, {
        "command": "ENHANCED_INSPECTION", "enable": req.enable, "camera_id": "CAM_01",
        "duration_s": ENHANCED_DEFAULT_DURATION, "reason": "手動下達巡檢", "timestamp": time.time(),
    }, qos=1)
    return {"status": "success"}


@app.get("/api/vision/sources")
async def get_sources():
    return {"current": vision_worker.source_spec, "sources": VisionWorker.list_sources()}


@app.post("/api/vision/source")
async def set_source(req: SourceRequest):
    valid = {s["id"] for s in VisionWorker.list_sources()}
    if req.source not in valid:
        raise HTTPException(400, f"未知的影像來源: {req.source}")
    vision_worker.request_source(req.source)
    return {"status": "success", "source": req.source}


@app.post("/api/vision/zone")
async def set_zone(req: ZoneRequest):
    try:
        vision_worker.set_zone(req.points)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"status": "success", "zone": vision_worker.zone_norm}


@app.get("/api/events")
async def get_events(limit: int = 50):
    return list(reversed(list(vision_worker.events)))[:limit]


@app.get("/api/rules")
async def get_rule_log():
    return rules.rule_log


# -------------------------------------------------------------
# 即時視訊 MJPEG 串流
# -------------------------------------------------------------
async def generate_mjpeg_stream():
    while True:
        frame_bytes = vision_worker.get_frame()
        if frame_bytes is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        await asyncio.sleep(0.04)


@app.get("/api/video_feed")
async def video_feed():
    return StreamingResponse(generate_mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


# -------------------------------------------------------------
# WebSocket
# -------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.add(websocket)
    await websocket.send_text(json.dumps({
        "type": "init_state",
        "data": {
            "mqtt_broker": mqtt_mgr.broker,
            "topic_root": TOPIC_ROOT,
            "thresholds": THRESHOLDS,
            "tower_light": sensor_sim.tower_light,
            "buzzer": sensor_sim.buzzer,
            "anomaly_mode": sensor_sim.anomaly_mode,
        },
    }))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_websockets.discard(websocket)


# -------------------------------------------------------------
# 靜態資源
# -------------------------------------------------------------
EVIDENCE_DIR.mkdir(exist_ok=True)
app.mount("/evidence", StaticFiles(directory=str(EVIDENCE_DIR)), name="evidence")

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
async def get_index():
    index_file = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "SafeFactory AIoT Backend is running. Frontend index.html not found."}
