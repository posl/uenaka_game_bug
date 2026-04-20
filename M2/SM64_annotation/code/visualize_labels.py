from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import io

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "../0Star/25/frame_label1.csv"
OUTPUT_PDF = SCRIPT_DIR / "../0Star/25/bug_visualize.pdf"

# ラベル列ごとに色を割り当て
LABEL_COLORS = [
    (0.2, 0.6, 1.0),   # blue
    (1.0, 0.4, 0.3),   # red
    (0.3, 0.85, 0.4),  # green
    (1.0, 0.75, 0.2),  # orange
    (0.7, 0.3, 0.9),   # purple
]


def get_segments(series: pd.Series) -> list[tuple[int, int]]:
    """ラベルが1の連続区間を (start_idx, end_idx) のリストで返す．"""
    segments = []
    in_seg = False
    seg_start = 0
    for i, val in enumerate(series):
        if val == 1 and not in_seg:
            in_seg = True
            seg_start = i
        elif val != 1 and in_seg:
            in_seg = False
            segments.append((seg_start, i - 1))
    if in_seg:
        segments.append((seg_start, len(series) - 1))
    return segments


def format_count(segments: list[tuple[int, int]]) -> str:
    """区間ごとの枚数を加算式の文字列で返す．"""
    counts = [end - start + 1 for start, end in segments]
    if len(counts) == 1:
        return str(counts[0])
    return " + ".join(map(str, counts)) + " = " + str(sum(counts))


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"{CSV_PATH} not found")

    df = pd.read_csv(CSV_PATH)

    # frame列を除いたラベル列を取得
    label_cols = [c for c in df.columns if c != "frame"]
    total_frames = len(df)

    # ---- 集計 & コンソール出力 ----
    label_segments: dict[str, list[tuple[int, int]]] = {}
    for col in label_cols:
        segs = get_segments(df[col])
        if segs:
            label_segments[col] = segs
            count_str = format_count(segs)
            print(f"{col}: {count_str} frames")

    if not label_segments:
        print("ラベルが立っているフレームはありません．")
        return

    # ---- matplotlib で可視化画像を生成（1行に全ラベルを重ねる）----
    active_labels = list(label_segments.keys())

    last_frame = total_frames - 1

    fig_width = 14
    fig_height = 1.2

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, last_frame)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("Frame index", fontsize=9)
    ax.set_yticks([])
    ax.grid(axis="x", linestyle="--", alpha=0.3, zorder=2)

    # 背景をまず全体灰色で塗る（ラベルなし区間の表現）
    ax.barh(y=0, width=total_frames, left=0, height=0.7,
            color=(0.8, 0.8, 0.8), edgecolor="none", zorder=1)

    # 各ラベルを同一行に重ねて描画
    for idx, col in enumerate(active_labels):
        color = LABEL_COLORS[idx % len(LABEL_COLORS)]
        display_name = col.removesuffix("_label")
        for start, end in label_segments[col]:
            ax.barh(
                y=0,
                width=end - start + 1,
                left=start,
                height=0.7,
                color=color,
                edgecolor="none",
                alpha=0.85,
                zorder=3,
                label=display_name if (start, end) == label_segments[col][0] else "_nolegend_",
            )

    # 凡例（重複エントリを除去，バー右外に配置）
    handles, labels_leg = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels_leg):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(),
              loc="center left", bbox_to_anchor=(1.03, 0.5),
              fontsize=8, framealpha=0.85, edgecolor="gray")

    # x軸：自動目盛り＋最終フレーム番号を追加
    ax.set_xlim(0, last_frame)
    auto_ticks = [t for t in ax.get_xticks() if 0 <= t <= last_frame]
    all_ticks = sorted(set(auto_ticks + [last_frame]))
    ax.set_xticks(all_ticks)
    ax.set_xticklabels([str(int(t)) for t in all_ticks], fontsize=8)

    plt.tight_layout()

    # PNG として一時保存（72dpi 換算で pt サイズが fig サイズと一致）
    png_path = SCRIPT_DIR / "_tmp_timeline.png"
    DPI = 150
    fig.savefig(str(png_path), dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # PNG の実ピクセルサイズを取得し，pt に換算してページサイズを決定
    from PIL import Image as PILImage
    with PILImage.open(png_path) as im:
        px_w, px_h = im.size                    # pixels
    pt_per_px = 72.0 / DPI
    img_w = px_w * pt_per_px
    img_h = px_h * pt_per_px

    margin = 10                                  # 上下左右の余白（pt）
    page_w = img_w + margin * 2
    page_h = img_h + margin * 2

    # ---- reportlab で PDF に貼り付け（visualize結果のみ）----
    c = canvas.Canvas(str(OUTPUT_PDF), pagesize=(page_w, page_h))
    c.drawImage(str(png_path), margin, margin, width=img_w, height=img_h)

    c.showPage()
    c.save()

    # 一時ファイル削除
    png_path.unlink(missing_ok=True)

    print(f"\n[INFO] Saved -> {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
