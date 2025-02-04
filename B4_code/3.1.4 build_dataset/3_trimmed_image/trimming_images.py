import cv2
import os
import pandas as pd

# CSVファイルのパス
csv_file = '/Users/uenakayuto/main-research/mycode/jikken/jikken3/trimming_data.csv'  # 必要に応じてCSVファイルのパスを設定

# トリミングとファイル保存の処理を行う関数
def trim_images(input_dir, output_dir, width, height, x, y, start_index, end_index):
    # 出力ディレクトリを作成（存在しない場合のみ）
    os.makedirs(output_dir, exist_ok=True)
    
    # ディレクトリ内の全てのファイルを取得
    for filename in sorted(os.listdir(input_dir)):
        # 画像ファイルのみを処理（例として、jpg, png, jpegの拡張子）
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # ファイル名の数字部分を抽出して番号を取得（例: 40.jpg -> 40）
            try:
                file_number = int(os.path.splitext(filename)[0])
            except ValueError:
                # 数字がないファイル名は無視
                continue
            
            # 指定範囲内の画像ファイルのみ処理
            if start_index <= file_number <= end_index:
                input_path = os.path.join(input_dir, filename)
                
                # 新しいファイル名は、(元の番号 - 開始番号).jpg
                new_filename = f"{file_number - start_index}.jpg"
                output_path = os.path.join(output_dir, new_filename)
                
                # 画像を読み込む
                image = cv2.imread(input_path)
                
                # 画像が正常に読み込まれたか確認
                if image is None:
                    print(f"画像を読み込めませんでした: {filename}")
                    continue
                
                # トリミング処理
                cropped_image = image[y:y+height, x:x+width]
                
                # トリミングされた画像を保存
                cv2.imwrite(output_path, cropped_image)
                
                # print(f"{filename} を {new_filename} に保存しました。")

# CSVファイルを読み込む
df = pd.read_csv(csv_file)

# 各行について処理を行う
for index, row in df.iterrows():
    # 各行から必要な値を取得
    folder_name = row['folder_name']
    width = int(row['width'])
    height = int(row['height'])
    x = int(row['x'])
    y = int(row['y'])
    start_index = int(row['start_index'])
    end_index = int(row['end_index'])
    
    # 各ディレクトリ内の'frames'フォルダを指定
    input_dir = os.path.join('/Users/uenakayuto/main-research/mycode/jikken/jikken3/dataset3', folder_name, 'frames')
    
    # 出力先となる'trimmed_frames'フォルダを同じディレクトリ内に作成
    output_dir = os.path.join('/Users/uenakayuto/main-research/mycode/jikken/jikken3/dataset3', folder_name, 'trimmed_frames')

    # トリミング処理を実行
    trim_images(input_dir, output_dir, width, height, x, y, start_index, end_index)

print("全ての画像のトリミングが完了しました。")