import os
import pandas as pd

# 初期化
directory = '/Users/uenakayuto/main-research/mycode/jikken/jikken1/dataset1'  # 解析するディレクトリのパス
file_types = ['train', 'validation', 'test']  # 対象となるテキストファイル名
label_counts = {ft: {'0': 0, '1': 0} for ft in file_types}  # 初期化

# ディレクトリの探索
for subdir in os.listdir(directory):
    subdir_path = os.path.join(directory, subdir)
    if os.path.isdir(subdir_path):
        # サブディレクトリ内の 'frame_label.csv' のパス
        csv_file = os.path.join(subdir_path, 'frame_label.csv')
        
        if os.path.exists(csv_file):
            # サブディレクトリ内のテキストファイルをチェック
            for file_type in file_types:
                txt_file = os.path.join(subdir_path, f'{file_type}')
                if os.path.exists(txt_file):
                    # CSVファイルを読み込み
                    df = pd.read_csv(csv_file)
                    # label列の値をカウントして集計
                    label_counts[file_type]['0'] += (df['label'] == 0).sum()
                    label_counts[file_type]['1'] += (df['label'] == 1).sum()

# 結果の出力
for file_type, counts in label_counts.items():
    print(f"ファイル名: {file_type}")
    print(f"  label=0 の数: {counts['0']}")
    print(f"  label=1 の数: {counts['1']}")