#!/usr/bin/env python3
import pandas as pd


def main() -> None:
    csv_path = "frame_label.csv"
    df = pd.read_csv(csv_path)

    label_cols = [c for c in df.columns if c.endswith("_label") and c != "bug_label"]

    total_frames = len(df)
    any_bug = df[label_cols].any(axis=1)
    any_bug_count = int(any_bug.sum())
    bug_label_count = int(df["bug_label"].sum()) if "bug_label" in df.columns else None

    per_label = df[label_cols].sum().astype(int).sort_values(ascending=False)

    multi_label_count = int((df[label_cols].sum(axis=1) >= 2).sum())
    max_labels = int(df[label_cols].sum(axis=1).max())

    # Pairwise overlaps (bug_label excluded via label_cols)
    pairs = []
    for i, a in enumerate(label_cols):
        for b in label_cols[i + 1 :]:
            overlap = int((df[a].astype(bool) & df[b].astype(bool)).sum())
            if overlap:
                pairs.append((a, b, overlap))
    pairs.sort(key=lambda x: -x[2])

    print("=== Frame Label Summary ===")
    print(f"Total frames: {total_frames}")
    print(f"Any bug (OR of categories): {any_bug_count}")
    if bug_label_count is not None:
        print(f"bug_label count: {bug_label_count}")
        print(f"bug_label equals OR: {bug_label_count == any_bug_count}")

    print("\n--- Counts per label ---")
    for col, cnt in per_label.items():
        print(f"{col:10s}: {cnt}")

    print("\n--- Overlap ---")
    print(f"Frames with >=2 categories: {multi_label_count}")
    print(f"Max categories on a single frame: {max_labels}")
    print("Pairwise overlaps (non-zero):")
    if pairs:
        for a, b, overlap in pairs:
            print(f"  {a} & {b}: {overlap}")
    else:
        print("  None")


if __name__ == "__main__":
    main()
