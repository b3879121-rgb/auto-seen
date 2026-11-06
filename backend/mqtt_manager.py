"""
MQTT 連線與通訊管理模組 (MQTT Client & Manager)
提供非同步事件回呼、自訂主題訂閱、LWT 遺囑機制及向外發布訊息能力。
"""

import json
import logging
import time
from typing import Callable, Dict, List
import paho.mqtt.client as mqtt

from backend.config import (
    MQTT_BROKER,
    MQTT_PORT,
    MQTT_CLIENT_ID,
    MQTT_KEEPALIVE,
    TOPIC_TELEMETRY_ENV,
    TOPIC_TELEMETRY_MACHINE,
    TOPIC_VISION_ALERTS,
    TOPIC_CONTROL_ALARM,
    TOPIC_CONTROL_CAMERA,
    TOPIC_DEVICE_STATUS
)

logger = logging.getLogger("MQTTManager")
logging.basicConfig(level=logging.INFO)

class MQTTManager:
    def __init__(self, client_id: str = MQTT_CLIENT_ID, broker: str = MQTT_BROKER, port: int = MQTT_PORT):
        self.broker = broker
        self.port = port
        self.client_id = f"{client_id}_{int(time.time())}"
        self.callbacks: List[Callable[[str, dict], None]] = []
        self.is_connected = False
        
        # 建立 Paho MQTT 客戶端 (支援 paho-mqtt v2.x 及 v1.x)
        try:
            self.client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=self.client_id
            )
        except AttributeError:
            self.client = mqtt.Client(client_id=self.client_id)
            
        self._setup_lwt()
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    def _setup_lwt(self):
        """設定遺囑訊息 (Last Will and Testament, LWT)"""
        lwt_payload = json.dumps({
            "device": self.client_id,
            "status": "offline",
            "reason": "unexpected_disconnect",
            "timestamp": time.time()
        })
        self.client.will_set(
            topic=TOPIC_DEVICE_STATUS,
            payload=lwt_payload,
            qos=1,
            retain=True
        )

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            self.is_connected = True
            logger.info(f"✅ 成功連線至 MQTT Broker: {self.broker}:{self.port}")
            
            # 發布上線狀態 (Retained Message)
            online_payload = json.dumps({
                "device": self.client_id,
                "status": "online",
                "timestamp": time.time()
            })
            self.client.publish(TOPIC_DEVICE_STATUS, online_payload, qos=1, retain=True)
            
            # 訂閱工廠所有相關主題
            topics = [
                (TOPIC_TELEMETRY_ENV, 0),
                (TOPIC_TELEMETRY_MACHINE, 0),
                (TOPIC_VISION_ALERTS, 1),
                (TOPIC_CONTROL_ALARM, 1),
                (TOPIC_CONTROL_CAMERA, 1),
                (TOPIC_DEVICE_STATUS, 1),
            ]
            self.client.subscribe(topics)
            logger.info("📡 已訂閱主題: " + ", ".join([t[0] for t in topics]))
        else:
            logger.error(f"❌ 連線失敗，代碼: {rc}")

    def _on_disconnect(self, client, userdata, rc, *args, **kwargs):
        self.is_connected = False
        logger.warning(f"⚠️ 與 MQTT Broker 中斷連線 (代碼: {rc})")

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8"))
        except Exception as e:
            logger.error(f"解析 MQTT 訊息失敗 ({msg.topic}): {e}")
            return
        self._dispatch(msg.topic, data)

    def _dispatch(self, topic: str, data: dict):
        # 廣播給註冊的監聽器 (規則引擎、虛擬塔燈、攝影機、WebSocket)
        for cb in self.callbacks:
            try:
                cb(topic, data)
            except Exception as e:
                logger.exception(f"回呼處理錯誤 ({topic}): {e}")

    def register_callback(self, callback: Callable[[str, dict], None]):
        """註冊新訊息接收回呼函數"""
        self.callbacks.append(callback)

    def publish(self, topic: str, data: dict, qos: int = 0, retain: bool = False):
        """發布 MQTT 訊息"""
        if not self.is_connected:
            # 離線備援：連不上 Broker 時直接在本機派發，讓展示不會因為網路斷線而停擺
            self._dispatch(topic, data)
            return
        self.client.publish(topic, json.dumps(data), qos=qos, retain=retain)

    def start(self):
        """在背景非同步執行連線迴圈 (斷線會自動重連)"""
        try:
            self.client.reconnect_delay_set(min_delay=1, max_delay=30)
            self.client.connect_async(self.broker, self.port, keepalive=MQTT_KEEPALIVE)
            self.client.loop_start()
            logger.info("🚀 MQTT 背景迴圈已啟動")
        except Exception as e:
            logger.error(f"無法連線至 MQTT Broker ({self.broker}:{self.port}): {e}")

    def stop(self):
        """停止 MQTT 迴圈並正常斷線"""
        try:
            offline_payload = json.dumps({
                "device": self.client_id,
                "status": "offline",
                "reason": "normal_shutdown",
                "timestamp": time.time()
            })
            self.client.publish(TOPIC_DEVICE_STATUS, offline_payload, qos=1, retain=True)
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("🛑 MQTT 已停止並斷線")
        except Exception as e:
            logger.error(f"停止 MQTT 失敗: {e}")
