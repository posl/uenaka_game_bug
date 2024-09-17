from contextlib import contextmanager
import os
import sys
import csv
import cv2
import glob
import natsort
import numpy as np
import pandas as pd
from skimage.transform import resize

# 作業ディレクトリを変更する関数
@contextmanager
def cwd(path):
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)

# 作業ディレクトリを指定
if len(sys.argv) < 2:
    print("フォルダを指定してください。")
    sys.exit()
else:
    folder_name = sys.argv[1]

absolute_path = os.path.abspath(folder_name) # 絶対パスに変換
with cwd(absolute_path): # 作業ディレクトリを指定
    list_name_feature = [f"f_{i}" for i in range(14)]
    header = list_name_feature
    with open("features.csv", "w", newline='') as cluster_file: # ファイルを開く
        cluster_writer = csv.writer(cluster_file)
        cluster_writer.writerow(header)
        for video_id in os.listdir(): # フォルダ内のファイルをループ
            if not os.path.isdir(video_id): # フォルダでない場合はスキップ
                continue
            with cwd(video_id): # 作業ディレクトリを変更
                for bug_id in os.listdir():
                    if not os.path.isdir(bug_id):
                        continue
                    with cwd(bug_id): 
                        bug_id_name = "-".join(bug_id.split("-")[:3])
                        segment_file = "segment.mp4"
                        print("Start frame extraction")
                        os.makedirs("frames", exist_ok=True)

                        video_capture = cv2.VideoCapture(segment_file)
                        success, image = video_capture.read()
                        count = 0
                        while success:
                            cv2.imwrite("frames/%d.jpg" % count, image)
                            success, image = video_capture.read()
                            count += 1

                        image_list = natsort.natsorted(glob.glob(os.path.join("frames", '*.jpg')))
                        num_image = len(image_list)
                        image_id1 = 0
                        image_id2 = 1
                        total_matrix = np.zeros((180, 320))

                        for image in range(num_image - 1):
                            first = cv2.imread(image_list[image_id1], cv2.IMREAD_GRAYSCALE)
                            second = cv2.imread(image_list[image_id2], cv2.IMREAD_GRAYSCALE)

                            first_resize = resize(first, (180, 320))
                            second_resize = resize(second, (180, 320))

                            total_matrix += (first_resize != second_resize).astype(int)
                            image_id1 += 1
                            image_id2 += 1

                        vector_total_matrix = total_matrix.flatten() / num_image
                        vector_total_matrix.sort()

                        max_value_hm = vector_total_matrix.max()
                        min_value_hm = vector_total_matrix.min()
                        median_hm = np.median(vector_total_matrix)
                        mean_hm = np.mean(vector_total_matrix)
                        first_quantile_hm = np.quantile(vector_total_matrix, 0.25)
                        third_quantile_hm = np.quantile(vector_total_matrix, 0.75)

                        header_similarity = ["id_frame_pair", "frame1", "frame2", "correlation"]
                        with open("correlation.csv", 'w', newline='') as csv_file:
                            writer = csv.writer(csv_file)
                            writer.writerow(header_similarity)

                            image_id_1 = 0
                            image_id_2 = 1

                            for image in range(num_image - 1):
                                first_image = cv2.imread(image_list[image_id_1])
                                second_image = cv2.imread(image_list[image_id_2])

                                hist_image_1 = cv2.calcHist([cv2.cvtColor(first_image, cv2.COLOR_BGR2HSV)], [0, 1], None, [50, 60], [0, 180, 0, 256])
                                hist_image_2 = cv2.calcHist([cv2.cvtColor(second_image, cv2.COLOR_BGR2HSV)], [0, 1], None, [50, 60], [0, 180, 0, 256])

                                cv2.normalize(hist_image_1, hist_image_1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                                cv2.normalize(hist_image_2, hist_image_2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                                correlation = cv2.compareHist(hist_image_1, hist_image_2, cv2.HISTCMP_CORREL)

                                writer.writerow([image, image_list[image_id_1], image_list[image_id_2], correlation])
                                image_id_1 += 1
                                image_id_2 += 1

                        similarity = pd.read_csv("correlation.csv")
                        abs_similarity_score = similarity['correlation'].abs()
                        
                        max_value_similarity = abs_similarity_score.max()
                        min_value_similarity = abs_similarity_score.min()
                        median_similarity = abs_similarity_score.median()
                        mean_similarity = abs_similarity_score.mean()
                        first_quantile_similarity = abs_similarity_score.quantile(0.25)
                        third_quantile_similarity = abs_similarity_score.quantile(0.75)

                        result = [
                            max_value_hm, min_value_hm, median_hm, mean_hm, first_quantile_hm, third_quantile_hm,
                            max_value_similarity, min_value_similarity, median_similarity, mean_similarity,
                            first_quantile_similarity, third_quantile_similarity, bug_id_name, video_id
                        ]

                        cluster_writer.writerow(result)