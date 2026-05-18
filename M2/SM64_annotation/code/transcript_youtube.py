import subprocess
import re
from pathlib import Path

URL = "https://youtu.be/oowWfsv5B1s"

OUTPUT_DIR = Path(
    "/Users/uenakayuto/main-research/uenaka_game_bug/M2/SM64_annotation/0Star/25"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# 字幕ダウンロード
# -------------------------------------------------
subprocess.run([
    "yt-dlp",
    "--write-auto-subs",
    "--sub-langs", "en",
    "--skip-download",
    "--sub-format", "vtt",
    "-o", str(OUTPUT_DIR / "%(title)s.%(ext)s"),
    URL
], check=True)

# -------------------------------------------------
# VTTファイル取得
# -------------------------------------------------
vtt_files = list(OUTPUT_DIR.glob("*.en.vtt"))

if not vtt_files:
    raise FileNotFoundError("No VTT file found")

vtt_file = vtt_files[0]

print("Using:", vtt_file)

# -------------------------------------------------
# 時刻処理
# -------------------------------------------------
def parse_time(t):

    t = t.strip()

    if t.count(":") == 2:
        h, m, s = t.split(":")
        sec, ms = s.split(".")
        return (
            int(h) * 3600
            + int(m) * 60
            + int(sec)
            + int(ms) / 1000
        )

    elif t.count(":") == 1:
        m, s = t.split(":")
        sec, ms = s.split(".")
        return (
            int(m) * 60
            + int(sec)
            + int(ms) / 1000
        )

    raise ValueError(t)

def format_time(sec):

    sec = max(sec, 0)

    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))

    return f"{h:02}:{m:02}:{s:02}.{ms:03}"

# -------------------------------------------------
# 字幕抽出
# -------------------------------------------------
with open(vtt_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

output = []

i = 0

while i < len(lines):

    line = lines[i].strip()

    if "-->" in line:

        try:
            parts = line.split(" --> ")

            start_str = parts[0].strip()
            end_str = parts[1].split()[0].strip()

            start = parse_time(start_str)
            end = parse_time(end_str)

        except Exception as e:
            print("Skip timing parse:", line)
            i += 1
            continue

        # 3〜377秒のみ
        if end >= 3 and start <= 377:

            new_start = start - 3
            new_end = end - 3

            text_lines = []

            i += 1

            while i < len(lines):

                txt = lines[i].strip()

                if not txt:
                    break

                # HTMLタグ削除
                txt = re.sub(r"<[^>]+>", "", txt)

                # 重複除去
                if txt and txt not in text_lines:
                    text_lines.append(txt)

                i += 1

            if text_lines:

                output.append(
                    f"{format_time(new_start)} --> {format_time(new_end)}\n"
                    + " ".join(text_lines)
                    + "\n"
                )

    i += 1

# -------------------------------------------------
# 保存
# -------------------------------------------------
output_file = OUTPUT_DIR / "transcript_trimmed.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print("Saved:", output_file)
