"""
SafeFactory AIoT — 一鍵啟動入口腳本 (Entrypoint)
自動檢查環境並啟動 FastAPI、MQTT、YOLO 與 Web 監控儀表板。
"""

import sys
import os

# 設定 Windows 控制台為 UTF-8 編碼避免 cp950 編碼問題
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("[SafeFactory AIoT] 智慧工廠工安監控系統正在啟動中...")
    print("[MQTT Broker] broker.emqx.io:1883 (公用測試 Broker)")
    print("[Dashboard URL] http://localhost:8000")
    print("=" * 60)
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
