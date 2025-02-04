import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import sys
import os
import ffmpeg
from pytubefix import YouTube  # pytubeの代わりにpytubefixを使用
import cv2

# 出力フォルダを指定
if len(sys.argv) < 2:
    print("出力フォルダを指定してください。")
    sys.exit()
else:
    output = sys.argv[1]

# 指定された出力フォルダを作成
os.makedirs(output, exist_ok=True)

# 認証の設定
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name(
    '/Users/uenakayuto/main-research/mycode/sheet_analysis/bug-check-sheet-analysis-6197113950c7.json', scope)  # 認証情報ファイルへのパスを指定
gc = gspread.authorize(credentials)

# スプレッドシートにアクセス
spreadsheet = gc.open_by_key('1kicnsdFcgDml2eBxJ7J6m431knn0UeEueOlGY-mr4wg')

# シート2のデータを取得
worksheet = spreadsheet.get_worksheet(1)  # シート2（インデックス1）
data = worksheet.get_all_values()

# pandasのデータフレームに変換
df = pd.DataFrame(data[1:], columns=data[0])  # シート2のデータ

# video_idごとにグループ化して処理を最適化
video_groups = df.groupby('video_id')

for video_id, group in video_groups:
    # video_idフォルダを作成
    video_folder = os.path.join(output, video_id)
    os.makedirs(video_folder, exist_ok=True)

    # 同じvideo_idに対しては1回だけ動画をダウンロード
    first_row = group.iloc[0]
    youtube_url = first_row['URL']  # YouTubeのリンク（時間指定のURL）

    # 動画を一度だけダウンロード
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(file_extension='mp4').first()
    temp_video = stream.download(filename=os.path.join(video_folder, 'video.mp4'))

    # 同じvideo_idに対してはrecord_startとrecord_endが同じなので1回だけ処理
    record_start = int(first_row['record_start'])  # record_start列から取得
    record_end = int(first_row['record_end'])  # record_end列から取得
    duration = record_end - record_start  # 動画の長さ

    # 切り抜き動画の保存先をtemp_videoと同じディレクトリにする
    segment_folder = video_folder  # 同じフォルダ内に切り抜き動画を保存
    segment_path = os.path.join(segment_folder, f"segment_{record_start}_{record_end}.mp4")

    # ffmpeg を使用して動画を切り抜き
    ffmpeg.input(temp_video, ss=record_start, t=duration).output(segment_path).run()

    # フレームを抽出して保存する（切り抜き動画から抽出）
    frames_folder = os.path.join(segment_folder, 'frames')
    os.makedirs(frames_folder, exist_ok=True)

    # OpenCVで切り抜き動画を読み込み、フレームを抽出
    cap = cv2.VideoCapture(segment_path)
    frame_count = 0
    success, frame = cap.read()

    while success:
        # フレームをJPEG形式で保存
        frame_path = os.path.join(frames_folder, f"{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)

        frame_count += 1
        success, frame = cap.read()

    cap.release()

    # 一時ファイルと切り抜き動画を削除
    os.remove(temp_video)  # 一時ファイルを削除
    os.remove(segment_path)  # 切り抜き動画を削除