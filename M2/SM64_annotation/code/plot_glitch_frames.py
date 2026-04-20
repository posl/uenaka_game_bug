"""
plot_glitch_frames.py
- 実行場所 : SM64_annotation/code/
- Input    : SM64_annotation/0Star/bug_stats.csv
- Output   : SM64_annotation/0Star/glitch_frames.pdf
"""

import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── 0. パス定義 ─────────────────────────────────────────────────
# __file__ = SM64_annotation/code/plot_glitch_frames.py
# .parent        = SM64_annotation/code/
# .parent.parent = SM64_annotation/
BASE_DIR = Path(__file__).parent.parent
CSV_PATH = BASE_DIR / "0Star" / "bug_stats.csv"
OUT_PATH = BASE_DIR / "0Star" / "glitch_frames.pdf"

# ── 1. データ読み込み ───────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

# ── 2. Record_XX カラムを抽出し，番号順にソート ─────────────────
record_cols = [c for c in df.columns if re.fullmatch(r"Record_\d+", c)]
record_cols.sort(key=lambda c: int(re.search(r"\d+", c).group()))

# ── 3. X 軸ラベル（"Record_01" → "01" など番号部分のみ）──────────
x_labels = [re.search(r"\d+", c).group() for c in record_cols]
x_pos = range(len(record_cols))

# ── 4. プロット ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

for _, row in df.iterrows():
    glitch = row["Glitch_label"]
    y_vals = [row[c] for c in record_cols]
    ax.plot(x_pos, y_vals, marker="o", label=glitch)

ax.set_xticks(list(x_pos))
ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
ax.xaxis.set_minor_locator(ticker.NullLocator())

ax.set_xlabel("Record", fontsize=12)
ax.set_ylabel("Num of Frames", fontsize=12)
ax.set_title(
    "Frame Count Transition of Each Glitch\nin SM64 0-Star Speedrun Categories",
    fontsize=13,
)
ax.legend(title="Glitch", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(OUT_PATH)
plt.show()
print(f"Saved: {OUT_PATH}")
