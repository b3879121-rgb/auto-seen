"""
SafeFactory 自動化展示腳本
先執行 python run.py 啟動系統，再另開終端機執行 python test_demo.py
"""

import json
import sys
import time
import urllib.request

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE = "http://localhost:8000"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=5) as r:
        return json.loads(r.read().decode())


def post(path, body=None):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(BASE + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode())


def show(title):
    s = get("/api/status")
    v = s["vision"]
    print(f"  {title}")
    print(f"    塔燈 {s['tower_light']} | 蜂鳴器 {'ON' if s['buzzer'] else 'OFF'} | 原因：{s['alarm_reason']}")
    print(f"    攝影機模式 {v['mode']} (剩 {v['enhanced_remaining']}s) | AI {v['infer_fps']} Hz | {v['device']}")
    return s


print("=" * 60)
print("SafeFactory AIoT 連動流程展示")
print("=" * 60)

print("\n[1] 初始狀態")
s = show("系統基線")
print(f"    MQTT {'已連線' if s['mqtt_connected'] else '離線 (本機備援)'} | 主題 {s['topic_root']}/#")
print(f"    影像來源 {s['vision']['source']} | PPE 模型 {'已載入' if s['vision']['ppe_model_loaded'] else '未載入'}")

print("\n[2] IoT → AI：注入高溫異常，等待連續 3 筆超標")
post("/api/control/anomaly", {"enable": True})
time.sleep(5)
s = show("連動結果")
ok = s["temp_high"] and s["vision"]["mode"] == "enhanced"
print(f"    {'✅' if ok else '❌'} 高溫觸發加強巡檢")

print("\n[3] 規則引擎觸發紀錄")
for r in get("/api/rules")[:5]:
    print(f"    [{r['kind']}] {r['text']}")

print("\n[4] 違規存證 (最近 5 筆)")
events = get("/api/events?limit=5")
if not events:
    print("    目前沒有違規事件")
for e in events:
    t = time.strftime("%H:%M:%S", time.localtime(e["timestamp"]))
    print(f"    {t} {e['event']:<16} {e['severity']:<8} {BASE}{e['snapshot_url']}")

print("\n[5] 蜂鳴器消音 → 手動復歸")
post("/api/control/buzzer_mute")
time.sleep(1)
show("消音後")
post("/api/control/alarm_reset")
time.sleep(2)
s = show("復歸後")
print(f"    {'✅' if s['tower_light'] == 'GREEN' and s['vision']['mode'] == 'normal' else '❌'} 回到綠燈與一般巡檢")

print("\n" + "=" * 60)
