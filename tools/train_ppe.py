"""
訓練安全帽 / 反光背心 (PPE) 偵測模型

使用 Ultralytics 內建的 Construction-PPE 資料集 (1000 張工地照片, CC BY 4.0)，
第一次執行會自動下載資料集與 yolo11n.pt 預訓練權重。
訓練完成後把最佳權重複製到 models/ppe_yolo11n.pt，系統啟動時會自動載入。

用法：
    .venv\\Scripts\\python.exe tools\\train_ppe.py            # 預設 60 epochs
    .venv\\Scripts\\python.exe tools\\train_ppe.py --epochs 100
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO("yolo11n.pt")
    model.train(
        data="construction-ppe.yaml",
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=0,
        workers=4,
        project=str(ROOT / "training"),
        name="ppe",
        exist_ok=True,
        patience=20,
        plots=True,
    )

    best = ROOT / "training" / "ppe" / "weights" / "best.pt"
    target = ROOT / "models" / "ppe_yolo11n.pt"
    target.parent.mkdir(exist_ok=True)
    shutil.copy(best, target)
    print(f"[DONE] PPE 模型已輸出至 {target}")


if __name__ == "__main__":
    main()
