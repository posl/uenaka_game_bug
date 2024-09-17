import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import sys
import os
import ffmpeg
from pytubefix import YouTube  # pytubeの代わりにpytubefixを使用
import re

# YouTubeのURLから秒数を抽出する関数
def extract_time_from_url(url):
    # URLのクエリパラメータに「t=」や「start=」が含まれている場合、時間を抽出
    time_match = re.search(r'(?:t=|start=)(\d+)', url)
    if time_match:
        return int(time_match.group(1))  # 秒数に変換
    return 0  # 見つからない場合は0秒を返す

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

# シート3のデータを取得
worksheet = spreadsheet.get_worksheet(2)  # シート3（インデックス2）
data = worksheet.get_all_values()

# pandasのデータフレームに変換
df = pd.DataFrame(data[1:], columns=data[0])  # シート3のデータ

# video_idごとにグループ化して処理を最適化
video_groups = df.groupby('video_id')

for video_id, group in video_groups:
    # video_idフォルダを作成
    video_folder = os.path.join(output, video_id)
    os.makedirs(video_folder, exist_ok=True)

    # 同じvideo_idに対しては1回だけ動画をダウンロード
    first_row = group.iloc[0]
    youtube_url = first_row['start_time']  # YouTubeのリンク（時間指定のURL）

    # 動画を一度だけダウンロード
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(file_extension='mp4').first()
    temp_video = stream.download(filename=os.path.join(video_folder, 'temp_video.mp4'))

    # 各bug_idごとに動画を切り抜き
    for _, row in group.iterrows():
        bug_id = row['bug_id']
        bug_url = row['start_time']  # 各bug_idごとに異なるYouTubeリンク（時間指定のURL）
        
        # URLから開始時間を抽出して秒数に変換
        start_time_seconds = extract_time_from_url(bug_url)
        duration = int(row['duration'])

        # フォルダ名にbug_idとURLから抽出した秒数を使用
        bug_folder_name = f"{bug_id}-{start_time_seconds}"
        bug_folder = os.path.join(video_folder, bug_folder_name)
        os.makedirs(bug_folder, exist_ok=True)

        # 開始時間と長さを指定して動画を切り抜き
        output_path = os.path.join(bug_folder, 'segment.mp4')
        ffmpeg.input(temp_video, ss=start_time_seconds, t=duration).output(output_path).run()

    # 一時ファイルの削除
    os.remove(temp_video)