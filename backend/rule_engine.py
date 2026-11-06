"""
工安連動規則引擎 (Rules Engine)

所有連動都走 MQTT，不直接呼叫裝置程式，和接上實體設備時的架構一模一樣：

  AI → IoT：YOLO 違規事件 (vision/alerts)  → 發布 control/alarm  → 警報塔燈 / 蜂鳴器
  IoT → AI：環境溫度過高 (telemetry/env)   → 發布 control/camera → 攝影機進入加強巡檢
  融合規則：環境異常期間又有人闖入禁區      → 升級為 EMERGENCY
"""

import logging
import time

from backend.config import (
    TOPIC_TELEMETRY_ENV,
    TOPIC_VISION_ALERTS,
    TOPIC_CONTROL_ALARM,
    TOPIC_CONTROL_CAMERA,
    THRESHOLDS,
    ENHANCED_DEFAULT_DURATION,
)

logger = logging.getLogger("RuleEngine")

LIGHT_RANK = {"GREEN": 0, "YELLOW": 1, "RED": 2}


class RuleEngine:
    def __init__(self, mqtt_manager, tower):
        self.mqtt = mqtt_manager
        self.tower = tower  # 只讀取目前燈號，用來避免降級 (紅燈不會被黃燈蓋掉)

        self.temp_high = False
        self.co2_high = False
        self._temp_over = 0
        self._temp_under = 0
        self._co2_over = 0
        self.rule_log = []  # 最近觸發的規則，給儀表板顯示

    # ---------------------------------------------------------
    def on_mqtt_message(self, topic: str, data: dict):
        if topic == TOPIC_VISION_ALERTS:
            self._on_vision_alert(data)
        elif topic == TOPIC_TELEMETRY_ENV:
            self._on_env(data)

    # ---------------------------------------------------------
    # AI → IoT
    # ---------------------------------------------------------
    def _on_vision_alert(self, alert: dict):
        event = alert.get("event")
        if event == "PATROL_SNAPSHOT":
            return  # 巡檢照只存證，不觸發警報

        if event == "ZONE_INTRUSION":
            if self.temp_high or self.co2_high:
                # 融合規則：環境已經異常，又有人進入危險區 → 最高等級
                self._raise("RED", True, "EMERGENCY：環境異常區域內有人員闖入！")
                self._record("融合規則", "環境異常 + 人員闖入 → EMERGENCY")
            else:
                self._raise("RED", True, "人員闖入危險禁區")
        elif event in ("NO_HELMET", "NO_VEST"):
            label = "未戴安全帽" if event == "NO_HELMET" else "未穿反光背心"
            self._raise("YELLOW", True, f"人員{label}")

    # ---------------------------------------------------------
    # IoT → AI
    # ---------------------------------------------------------
    def _on_env(self, data: dict):
        temp = float(data.get("temperature", 0))
        co2 = float(data.get("co2", 0))
        need = THRESHOLDS["consecutive_readings"]
        t_max = THRESHOLDS["temperature_max"]
        t_clear = t_max - THRESHOLDS["temperature_hysteresis"]

        # 溫度：連續 N 筆超標才觸發；降到 (上限 - 遲滯) 以下連續 N 筆才解除
        self._temp_over = self._temp_over + 1 if temp > t_max else 0
        self._temp_under = self._temp_under + 1 if temp < t_clear else 0

        if not self.temp_high and self._temp_over >= need:
            self.temp_high = True
            reason = f"環境溫度過高 ({temp:.1f}°C > {t_max}°C)"
            self._raise("RED", True, reason)
            self.mqtt.publish(TOPIC_CONTROL_CAMERA, {
                "command": "ENHANCED_INSPECTION",
                "enable": True,
                "camera_id": "CAM_01",
                "zone": "ZONE_A",
                "duration_s": ENHANCED_DEFAULT_DURATION,
                "reason": reason,
                "timestamp": time.time(),
            }, qos=1)
            self._record("IoT→AI", f"{reason} → 攝影機加強巡檢")
            logger.warning(f"🔥 {reason}，已發布加強巡檢指令")

        elif self.temp_high and self._temp_under >= need:
            self.temp_high = False
            self.mqtt.publish(TOPIC_CONTROL_CAMERA, {
                "command": "ENHANCED_INSPECTION",
                "enable": False,
                "camera_id": "CAM_01",
                "reason": f"溫度恢復正常 ({temp:.1f}°C)",
                "timestamp": time.time(),
            }, qos=1)
            self._record("IoT→AI", f"溫度恢復 {temp:.1f}°C → 解除加強巡檢")

        # CO2：超標亮黃燈
        self._co2_over = self._co2_over + 1 if co2 > THRESHOLDS["gas_co2_max"] else 0
        if not self.co2_high and self._co2_over >= need:
            self.co2_high = True
            self._raise("YELLOW", False, f"CO2 濃度偏高 ({co2:.0f} ppm)")
        elif self.co2_high and co2 < THRESHOLDS["gas_co2_max"] * 0.9:
            self.co2_high = False

    # ---------------------------------------------------------
    def _raise(self, light: str, buzzer: bool, reason: str):
        """發布警報指令；目前已是更高等級的燈號時不降級。"""
        current = getattr(self.tower, "tower_light", "GREEN")
        if LIGHT_RANK.get(current, 0) > LIGHT_RANK[light]:
            light = current
        self.mqtt.publish(TOPIC_CONTROL_ALARM, {
            "command": "SET",
            "tower_light": light,
            "buzzer": buzzer,
            "reason": reason,
            "timestamp": time.time(),
        }, qos=1)
        self._record("AI→IoT" if "人員" in reason else "IoT", f"{reason} → 塔燈 {light}")

    def _record(self, kind: str, text: str):
        self.rule_log.insert(0, {"time": time.time(), "kind": kind, "text": text})
        del self.rule_log[50:]

    def reset(self):
        """手動復歸：清除條件狀態，若異常仍在會在 N 筆讀值後重新觸發。"""
        self.temp_high = self.co2_high = False
        self._temp_over = self._temp_under = self._co2_over = 0
