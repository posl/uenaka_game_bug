import json
import pandas as pd
from pathlib import Path

def calculate_metrics(file_path):
    results = []
    path = Path(file_path)
    
    if not path.exists():
        print(f"Error: File {file_path} not found.")
        return

    # JSONLファイルの読み込み
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    df = pd.DataFrame(results)

    # 混同行列の要素を定義
    # TP: バグ(1)を正しくバグ(1)と判定
    # FP: 正常(0)を誤ってバグ(1)と判定
    # FN: バグ(1)を誤って正常(0)と判定
    # TN: 正常(0)を正しく正常(0)と判定
    tp = len(df[(df['label'] == 1) & (df['prediction'] == 1)])
    fp = len(df[(df['label'] == 0) & (df['prediction'] == 1)])
    fn = len(df[(df['label'] == 1) & (df['prediction'] == 0)])
    tn = len(df[(df['label'] == 0) & (df['prediction'] == 0)])

    # 指標の計算
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # 結果の表示
    print(f"--- Results for: {path.name} ---")
    print(f"Samples: {len(df)}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print("-" * 30)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("-" * 30)
    print(f"Total Cost: ${df['cost'].sum():.4f}")

if __name__ == "__main__":
    # 集計したいファイルのパスを指定
    calculate_metrics('results/oob_vision_results.jsonl')
