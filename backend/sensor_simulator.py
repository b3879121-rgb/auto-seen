"""
工廠虛擬感測器與邊緣設備模擬器 (Virtual Sensor Simulator)
模擬實體 ESP32 / 邊緣閘道器定期發送環境遙測與機台振動數據，
並接收 MQTT 反向控制指令 (警報塔燈切換、蜂鳴器消音)。
"""

import json
import logging
import math
import random
import threading
import time
from backend.config import (
    TOPIC_TELEMETRY_ENV,
    TOPIC_TELEMETRY_MACHINE,
    TOPIC_CONTROL_ALARM,
    THRESHOLDS
)

logger = logging.getLogger("SensorSimulator")

class SensorSimulator:
    def __init__(self, mqtt_manager):
        self.mqtt = mqtt_manager
        self.running = False
        self.thread = None
        
        # 初始基準數據 (Base Values)
        self.temp_base = 26.5
        self.humidity_base = 60.0
        self.co2_base = 550.0
        self.vibration_base = 2.2
        self.rpm_base = 1750
        
        # 警報塔燈狀態 (GREEN, YELLOW, RED) 與蜂鳴器
        self.tower_light = "GREEN"
        self.buzzer = False
        self.last_alarm_reason = "Normal"
        
        # 異常注入標記 (供前端手動測試高溫或異常振動)
        self.anomaly_mode = False
        self.step = 0

    def trigger_anomaly(self, enable: bool = True):
        """動態注入異常 (供展示用)"""
        self.anomaly_mode = enable
        logger.info(f"🚨 感測器異常模擬模式: {'開啟' if enable else '關閉'}")

    def update_alarm_state(self, status: str, buzzer: bool, reason: str = ""):
        self.tower_light = status
        self.buzzer = buzzer
        if reason:
            self.last_alarm_reason = reason

    def on_mqtt_message(self, topic: str, data: dict):
        """
        虛擬警報塔燈 / 蜂鳴器：和實體 ESP32 一樣，只聽 MQTT 指令做動作。
        指令格式：
          {"command": "SET", "tower_light": "RED", "buzzer": true, "reason": "..."}
          {"command": "MUTE_BUZZER"}
          {"command": "RESET_ALARM"}
        """
        if topic != TOPIC_CONTROL_ALARM:
            return
        cmd = data.get("command")
        if cmd == "SET":
            self.update_alarm_state(data.get("tower_light", self.tower_light),
                                    bool(data.get("buzzer", self.buzzer)),
                                    data.get("reason", ""))
            logger.info(f"🚥 塔燈指令: {self.tower_light} / 蜂鳴器 {'ON' if self.buzzer else 'OFF'} ({self.last_alarm_reason})")
        elif cmd == "MUTE_BUZZER":
            self.buzzer = False
        elif cmd == "RESET_ALARM":
            self.update_alarm_state("GREEN", False, "Normal")

    def run_loop(self):
        logger.info("📡 虛擬 IoT 感測器模擬線程已啟動")
        while self.running:
            try:
                self.step += 1
                t = self.step * 0.1
                
                # 1. 產生環境數值 (平滑波動 + 異常加成)
                noise_temp = 0.5 * math.sin(t * 0.5) + random.uniform(-0.2, 0.2)
                noise_co2 = 25 * math.cos(t * 0.3) + random.uniform(-10, 10)
                
                curr_temp = self.temp_base + noise_temp
                curr_co2 = self.co2_base + noise_co2
                curr_tvoc = 150 + random.randint(-15, 20)
                curr_hum = self.humidity_base + 2.0 * math.sin(t * 0.2)
                
                # 2. 產生馬達機台振動數值
                vib_drift = 0.4 * math.sin(t * 1.2) + random.uniform(-0.1, 0.1)
                curr_vib = self.vibration_base + vib_drift
                curr_rpm = self.rpm_base + random.randint(-10, 10)
                
                # 異常注入加成
                if self.anomaly_mode:
                    curr_temp += 15.0  # 飆高溫 > 40°C
                    curr_vib += 6.5   # 振動劇烈 > 8.0 mm/s
                    curr_co2 += 750   # CO2 超標 > 1300 ppm

                # 3. 封裝環境 Payload
                env_payload = {
                    "timestamp": time.time(),
                    "device_id": "ENV_STATION_01",
                    "temperature": round(curr_temp, 2),
                    "humidity": round(curr_hum, 1),
                    "co2": round(curr_co2, 1),
                    "tvoc": round(curr_tvoc, 1),
                    "tower_light": self.tower_light,
                    "buzzer": self.buzzer,
                    "alarm_reason": self.last_alarm_reason
                }
                self.mqtt.publish(TOPIC_TELEMETRY_ENV, env_payload, qos=0)
                
                # 4. 封裝機台 Payload
                machine_payload = {
                    "timestamp": time.time(),
                    "device_id": "CNC_MOTOR_03",
                    "vibration": round(curr_vib, 2),
                    "rpm": curr_rpm,
                    "motor_temp": round(curr_temp + 12.0, 1),
                    "status": "WARNING" if curr_vib > THRESHOLDS["vibration_max"] else "OK"
                }
                self.mqtt.publish(TOPIC_TELEMETRY_MACHINE, machine_payload, qos=0)
                
                # 週期 1 秒發送一次
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"感測器模擬生成錯誤: {e}")
                time.sleep(1.0)

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.run_loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
            logger.info("🛑 虛擬 IoT 感測器已停止")
