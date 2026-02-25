#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import pandas as pd


def build_balanced(df: pd.DataFrame, label_col: str, seed: int) -> pd.DataFrame:
    bug_df = df[df[label_col] == 1]
    nonbug_df = df[df[label_col] == 0]

    bug_count = len(bug_df)
    if bug_count == 0:
        raise ValueError(f"No bug frames found for {label_col}.")
    if len(nonbug_df) < bug_count:
        raise ValueError(
            f"Not enough non-bug frames for {label_col}: "
            f"{len(nonbug_df)} < {bug_count}"
        )

    nonbug_sample = nonbug_df.sample(n=bug_count, random_state=seed)
    balanced = pd.concat([bug_df, nonbug_sample], ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced


def save_dataset(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build balanced bug/non-bug datasets from frame_label.csv"
    )
    parser.add_argument(
        "--csv",
        default="frame_label.csv",
        help="Path to frame_label.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="balanced_datasets",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    scenarios = {
        "all": "bug_label",
        "oob": "OoB_label",
        "fg": "FG_label",
    }

    out_dir = Path(args.out_dir)

    for name, col in scenarios.items():
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

        balanced = build_balanced(df[["frame", col]].rename(columns={col: "label"}), "label", args.seed)
        out_path = out_dir / f"{name}_balanced.csv.gz"
        save_dataset(balanced, out_path)

        bug_count = int((balanced["label"] == 1).sum())
        nonbug_count = int((balanced["label"] == 0).sum())
        print(f"{name}: bug={bug_count}, nonbug={nonbug_count}, total={len(balanced)} -> {out_path}")


if __name__ == "__main__":
    main()
