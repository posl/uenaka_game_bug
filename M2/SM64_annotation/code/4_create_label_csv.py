from pathlib import Path
import pandas as pd
from module.module import parse_multi_value

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "code" / "urls.csv"
GLITCH_LIST_ROOT = Path(__file__).resolve().parents[2]
GLITCH_LIST_PATH = GLITCH_LIST_ROOT / "SM64_glitch_list" / "glitch_list.json"

def main():

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"{CSV_PATH} not found")

    df = pd.read_csv(CSV_PATH)

    label_columns = [
        "BLJ_label",
        "LBLJ_label",
        "SBLJ_label",
        "DDD_skip_label",
        "Bob-omb_clip_label",
        "Cannonless_label",
        "Ship_clip_label",
        "Door_clip_label",
        "Box_TJ_label",
        "BLJ_strat_label",
        "Pillarless_label",
        "Insta_clip_label",
        "Pole_glitch_label",
        "Ground_pound_clipping_label",
        "Ledge_clipping_label",
        "Miscellaneous_wall_clipping_label",
        "Chuckya_clip_label",
        "MIPS_clip_label",
        "Monkey_bug_label"
    ]

    for _, row in df.iterrows():

        category = row["Category"].strip()
        num_id = str(row["No"]).strip()
        data_dir = PROJECT_ROOT / category / num_id

        if not data_dir.exists():
            print(f"[WARN] {data_dir} not found, skip")
            continue

        start_seconds = parse_multi_value(row["Start_sec"])
        end_seconds = parse_multi_value(row["End_sec"])

        if not (len(start_seconds) == len(end_seconds)):
            print(f"[ERROR] Mismatch in counts for {category}/{num_id}: {len(start_seconds)} != {len(end_seconds)}")
            continue

        for idx in range(len(start_seconds)):

            start_sec = int(start_seconds[idx])
            end_sec = int(end_seconds[idx])

            num_frames = int((end_sec - start_sec) * 30)

            frames = [
                f"{i}.jpg"
                for i in range(num_frames)
            ]

            out_df = pd.DataFrame({"frame": frames})

            for col in label_columns:
                out_df[col] = 0

            out_path = data_dir / f"frame_label{idx + 1}.csv"
            out_df.to_csv(out_path, index=False)

            print(f"[INFO] Created label CSV: {out_path}")

if __name__ == "__main__":
    main()
