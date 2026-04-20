"""
create_gif.py

SM64_annotation/0Star/25/test_41_frames/test_frames.csv の frame カラムを読み込み，
各フレーム xxxx.jpg に対して
  SM64_annotation/0Star/25/frames1/(xxxx-20).jpg 〜 (xxxx+20).jpg
の連番画像から GIF を作成し，
  SM64_annotation/0Star/25/test_41_frames/gif/xxxx.gif
として保存する．
"""

import csv
from pathlib import Path
from PIL import Image

# ── パス設定 ──────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent / "0Star" / "25"
CSV_PATH    = BASE_DIR / "test_41_frames/test_frames.csv"
FRAMES_DIR  = BASE_DIR / "frames1"
GIF_DIR     = BASE_DIR / "test_41_frames/gif"

# ── GIF パラメータ ────────────────────────────────────────
HALF_WINDOW  = 20          # 前後コマ数（xxxx ± 20 → 合計 41 枚）
FRAME_DURATION_MS = 50     # 1 フレームあたりの表示時間 [ms]（= 20 fps 相当）


def load_frames(center: int) -> list[Image.Image]:
    """center フレームを中心に ±HALF_WINDOW 枚の画像を読み込む．
    存在しないファイルはスキップする．"""
    images = []
    for idx in range(center - HALF_WINDOW, center + HALF_WINDOW + 1):
        if idx < 0:
            continue
        img_path = FRAMES_DIR / f"{idx:04d}.jpg"
        if not img_path.exists():
            print(f"  [WARN] 存在しないフレームをスキップ: {img_path}")
            continue
        images.append(Image.open(img_path).convert("RGB"))
    return images


def save_gif(images: list[Image.Image], out_path: Path) -> None:
    """画像リストを GIF として保存する．"""
    if not images:
        print(f"  [SKIP] 画像が 0 枚のため GIF を作成しない: {out_path.name}")
        return
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=FRAME_DURATION_MS,
        loop=0,          # 0 = 無限ループ
        optimize=False,
    )
    print(f"  [OK]   {out_path}  ({len(images)} frames)")


def main() -> None:
    GIF_DIR.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"処理対象: {len(rows)} 行")

    for row in rows:
        # "xxxx.jpg" → 整数 xxxx を取得
        frame_name = row["frame"].strip()          # e.g. "0042.jpg"
        stem       = Path(frame_name).stem         # e.g. "0042"
        center     = int(stem)                     # e.g. 42

        out_path = GIF_DIR / f"{stem}.gif"

        print(f"[{frame_name}] center={center}  →  {out_path.name}")
        images = load_frames(center)
        save_gif(images, out_path)

    print("完了．")


if __name__ == "__main__":
    main()
