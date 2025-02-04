import os
import pandas as pd
from collections import defaultdict

# 初期化
directory = '/Users/uenakayuto/main-research/mycode/jikken/jikken1/dataset1'  # 解析するディレクトリのパス
label_counts = defaultdict(lambda: {'0': 0, '1': 0})  # 初期値0の辞書

# ディレクトリの探索
for subdir in os.listdir(directory):
    subdir_path = os.path.join(directory, subdir)
    if os.path.isdir(subdir_path):
        # サブディレクトリ名の最初の数字を抽出
        first_number = subdir.split('-')[0]
        csv_file = os.path.join(subdir_path, 'frame_label.csv')
        
        if os.path.exists(csv_file):
            # CSVファイルを読み込み
            df = pd.read_csv(csv_file)
            # label列の値をカウント
            label_counts[first_number]['0'] += (df['label'] == 0).sum()
            label_counts[first_number]['1'] += (df['label'] == 1).sum()

# 結果の出力
for key, counts in sorted(label_counts.items()):
    print(f"サブディレクトリ名の最初の数字: {key}")
    print(f"  label=0 の数: {counts['0']}")
    print(f"  label=1 の数: {counts['1']}")