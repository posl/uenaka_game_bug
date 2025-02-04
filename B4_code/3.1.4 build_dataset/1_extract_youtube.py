import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

# スコープと認証の設定
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name('/Users/uenakayuto/main-research/mycode/sheet_analysis/bug-check-sheet-analysis-6197113950c7.json', scope)
gc = gspread.authorize(credentials)

# スプレッドシートにアクセス
spreadsheet = gc.open_by_key('1kicnsdFcgDml2eBxJ7J6m431knn0UeEueOlGY-mr4wg')

# シート1（インデックス 0）のデータ取得
worksheet1 = spreadsheet.get_worksheet(0)
data1 = worksheet1.get_all_values()

# pandasのデータフレームに変換
df1 = pd.DataFrame(data1[1:], columns=data1[0])

# URL列の名前が "URL" であると仮定してフィルタリング処理
df1 = df1[df1['URL'].str.startswith(('https://www.youtube.com/', 'https://youtu.be/'))]

# シート2（インデックス 1）に保存
worksheet2 = spreadsheet.get_worksheet(1)
worksheet2.clear()  # 既存のデータをクリア
worksheet2.update([df1.columns.values.tolist()] + df1.values.tolist())  # 更新

print("YouTubeリンク以外の行を削除し、シート2を更新しました。")