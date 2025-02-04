# 手動で良いトリミング範囲を見つけるためのプログラム
import cv2

# 入力ファイル（画像）
input_image = '/Users/uenakayuto/main-research/mycode/jikken/jikken3/dataset3/TotK/frames/18000.jpg'

# 出力ファイル（トリミングされた画像）
output_image = '/Users/uenakayuto/main-research/mycode/bug_clustering/make_dataset/trial_image.jpg'

# トリミング範囲の指定
width = 542   # トリミング後の幅
height = 302  # トリミング後の高さ
x = 98       # トリミング開始のx座標
y = 29         # トリミング開始のy座標

# 画像を読み込む
image = cv2.imread(input_image)

# トリミング処理
cropped_image = image[y:y+height, x:x+width]

# トリミングされた画像を保存
cv2.imwrite(output_image, cropped_image)

print("トリミングが完了しました。出力ファイル:", output_image)