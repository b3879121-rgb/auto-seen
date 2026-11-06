"""
中文文字繪製 (Chinese Text Renderer)

OpenCV 的 cv2.putText 只支援英文字，中文會變成問號。
這裡改用 Pillow + 系統中文字型繪製：每一幀先把要寫的字收集起來，
最後一次轉成 Pillow 影像畫完再轉回 OpenCV，避免反覆轉換拖慢速度。
"""

import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 依序嘗試：Windows 微軟正黑體 → macOS 蘋方 → Linux Noto CJK
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msjhbd.ttc",
    r"C:\Windows\Fonts\msjh.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


@lru_cache(maxsize=16)
def get_font(size: int):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)  # 找不到中文字型時退回預設字型 (中文會顯示不出來)


def bgr2rgb(c):
    return (c[2], c[1], c[0])


class TextLayer:
    """收集一幀要寫的文字，最後 render() 一次畫上去。"""

    def __init__(self):
        self.items = []

    def add(self, text, xy, color=(255, 255, 255), size=16, bg=None, pad=3, anchor="la"):
        """
        xy：文字位置；anchor 用 Pillow 規則 (la=左上、lb=左下、mm=置中)
        bg：背景色 (BGR)，有給就會先畫一塊底色，讓字在任何畫面上都清楚
        """
        self.items.append((text, xy, color, size, bg, pad, anchor))

    def render(self, img: np.ndarray) -> np.ndarray:
        if not self.items:
            return img
        pil = Image.fromarray(img[:, :, ::-1])
        draw = ImageDraw.Draw(pil)
        for text, (x, y), color, size, bg, pad, anchor in self.items:
            font = get_font(size)
            if bg is not None:
                l, t, r, b = draw.textbbox((x, y), text, font=font, anchor=anchor)
                draw.rectangle((l - pad, t - pad, r + pad, b + pad), fill=bgr2rgb(bg))
            draw.text((x, y), text, font=font, fill=bgr2rgb(color), anchor=anchor)
        return np.asarray(pil)[:, :, ::-1].copy()
