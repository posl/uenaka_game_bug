# import os
# import pandas as pd
# import cv2
# import cv2 as cv
# import numpy as np
# import glob
# import natsort
# import csv
# import statistics
# from skimage.transform import resize

# SAVE_PATH = "video"

# def youtubeVideo(path):
#     with open("features_frame_extraction.arff", "w") as arff_file:#, open("info.txt", "w") as info_file:
#         arff_file.write('''@RELATION classification
# @ATTRIBUTE maxheatmap REAL
# @ATTRIBUTE minheatmap REAL
# @ATTRIBUTE medheatmap REAL
# @ATTRIBUTE avgheatmap REAL
# @ATTRIBUTE 1qheatmap REAL
# @ATTRIBUTE 3qheatmap REAL
# @ATTRIBUTE maxsimilarity REAL
# @ATTRIBUTE minsimilarity REAL
# @ATTRIBUTE medsimilarity REAL
# @ATTRIBUTE avgsimilarity REAL
# @ATTRIBUTE 1qsimilarity REAL
# @ATTRIBUTE 3qsimilarity REAL
# @ATTRIBUTE classification {kw, kn, kwkn}
# @DATA 
# ''')

#         video_list = pd.read_csv(path)
#         downloaded_videos = {}

#         for idx, row in video_list.iterrows():
#             label = row['labeled']

#             path_video = f"segment-{idx}-{label}"
#             if not os.path.exists(path_video):
#                 os.mkdir(path_video)

#             if url not in downloaded_videos:
#                 try:
#                     my_video = YouTube(url)
#                 except:
#                     print("Connection Error")
#                     continue

#                 print("********************Download video*************************")
#                 # Download video
#                 try:
#                     my_video = my_video.streams.get_highest_resolution()
#                     video_path = my_video.download(SAVE_PATH, f"video-{idx}.mp4")
#                     downloaded_videos[url] = video_path
#                 except:
#                     with open("info.txt", "w") as f:
#                         f.write(url)
#                     continue
#             else:
#                 video_path = downloaded_videos[url]

#             start_segment = row['start-segment(s)']
#             end_segment = row['end-segment(s)']

#             in_file = ffmpeg.input(video_path)
#             segment_file = f"{path_video}/segment.mp4"

#             try:
#                 (
#                     in_file.trim(start=start_segment, end=end_segment).setpts('PTS-STARTPTS')
#                     .output(segment_file)
#                     .run()
#                 )
#             except:
#                 print("segment no-analysed")
#                 continue

#             print("Start frame extraction")
#             os.mkdir(f"{path_video}/frames")
#             try:
#                 video_capture = cv2.VideoCapture(segment_file)
#                 success, image = video_capture.read()
#                 count = 0
#                 while success:
#                     cv2.imwrite(f"{path_video}/frames/{count}.jpg", image)
#                     success, image = video_capture.read()
#                     count += 1
#             except:
#                 continue

#             image_list = list(glob.glob(os.path.join(f"{path_video}/frames", '*.jpg')))
#             order_image_list = natsort.natsorted(image_list)
#             num_image = len(image_list)

#             total_matrix = np.zeros((180, 320))
#             for image in range(num_image - 1):
#                 image1 = order_image_list[image]
#                 image2 = order_image_list[image + 1]

#                 first = cv.imread(image1)
#                 second = cv.imread(image2)

#                 first_gray = cv.cvtColor(first, cv.COLOR_BGR2GRAY)
#                 second_gray = cv.cvtColor(second, cv.COLOR_BGR2GRAY)

#                 first_resize = resize(first_gray, (180, 320))
#                 second_resize = resize(second_gray, (180, 320))

#                 array_frame1 = np.array(first_resize)
#                 array_frame2 = np.array(second_resize)

#                 check_frames = array_frame1 == array_frame2
#                 for i in range(len(check_frames)):
#                     for j in range(len(check_frames[i])):
#                         if not check_frames[i][j]:
#                             total_matrix[i][j] += 1

#             vector_total_matrix = total_matrix.flatten()
#             vector_total_matrix.sort()
#             max_value_hm = max(vector_total_matrix / num_image)
#             min_value_hm = min(vector_total_matrix / num_image)
#             median_hm = statistics.median(vector_total_matrix / num_image)
#             mean_hm = statistics.mean(vector_total_matrix / num_image)
#             first_quantile_hm = np.quantile(vector_total_matrix / num_image, 0.25)
#             third_quantile_hm = np.quantile(vector_total_matrix / num_image, 0.75)

#             header = ["id_frame_pair", "frame1", "frame2", "correlation"]
#             with open(f"{path_video}/segment-results.csv", 'w', newline='') as csv_file:
#                 writer = csv.writer(csv_file)
#                 writer.writerow(header)
#                 for image in range(num_image - 1):
#                     image1 = order_image_list[image]
#                     image2 = order_image_list[image + 1]

#                     first_image = cv.imread(image1)
#                     second_image = cv.imread(image2)

#                     hsv_image_1 = cv.cvtColor(first_image, cv.COLOR_BGR2HSV)
#                     hsv_image_2 = cv.cvtColor(second_image, cv.COLOR_BGR2HSV)

#                     h_bins = 50
#                     s_bins = 60
#                     histSize = [h_bins, s_bins]
#                     h_ranges = [0, 180]
#                     s_ranges = [0, 256]
#                     ranges = h_ranges + s_ranges
#                     channels = [0, 1]

#                     hist_image_1 = cv.calcHist([hsv_image_1], channels, None, histSize, ranges, accumulate=False)
#                     cv.normalize(hist_image_1, hist_image_1, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

#                     hist_image_2 = cv.calcHist([hsv_image_2], channels, None, histSize, ranges, accumulate=False)
#                     cv.normalize(hist_image_2, hist_image_2, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

#                     compare_method = cv.HISTCMP_CORREL
#                     correlation = cv.compareHist(hist_image_1, hist_image_2, compare_method)

#                     writer.writerow([image, image1, image2, correlation])

#             similarity = pd.read_csv(f"{path_video}/segment-results.csv")
#             similarity_score = similarity.correlation.abs()

#             max_value_similarity = max(similarity_score)
#             min_value_similarity = min(similarity_score)
#             median_similarity = statistics.median(similarity_score)
#             mean_similarity = statistics.mean(similarity_score)
#             first_quantile_similarity = np.quantile(similarity_score, 0.25)
#             third_quantile_similarity = np.quantile(similarity_score, 0.75)

#             data = (max_value_hm, min_value_hm, median_hm, mean_hm, first_quantile_hm, third_quantile_hm,
#                     max_value_similarity, min_value_similarity, median_similarity, mean_similarity,
#                     first_quantile_similarity, third_quantile_similarity, label)

#             sentence = ",".join(map(str, data))
#             if sentence:
#                 arff_file.write(sentence + '\n')

import os
import pandas as pd
import cv2
import numpy as np
import glob
import natsort
import csv
import statistics
from skimage.transform import resize

def youtubeVideo(path):
    with open("features_frame_extraction.arff", "w") as arff_file:
        arff_file.write('''@RELATION classification
@ATTRIBUTE maxheatmap REAL
@ATTRIBUTE minheatmap REAL
@ATTRIBUTE medheatmap REAL
@ATTRIBUTE avgheatmap REAL
@ATTRIBUTE 1qheatmap REAL
@ATTRIBUTE 3qheatmap REAL
@ATTRIBUTE maxsimilarity REAL
@ATTRIBUTE minsimilarity REAL
@ATTRIBUTE medsimilarity REAL
@ATTRIBUTE avgsimilarity REAL
@ATTRIBUTE 1qsimilarity REAL
@ATTRIBUTE 3qsimilarity REAL
@ATTRIBUTE classification {presentation, logic, balance, performance, non-informative}
@DATA 
''')

        video_list = pd.read_csv(path)

        for idx, row in video_list.iterrows():
            label = row['labeled']
            start= row['start-segment(s)']
            matching_keyword = row['matching-word']

            path_video = f"s-{start}-{matching_keyword}"

            # Frame extraction
            os.mkdir(f"{path_video}/frames")
            segment_file = f"{path_video}/segment.mp4"

            try:
                video_capture = cv2.VideoCapture(segment_file)
                success, image = video_capture.read()
                count = 0
                while success:
                    cv2.imwrite(f"{path_video}/frames/{count}.jpg", image)
                    success, image = video_capture.read()
                    count += 1
            except:
                continue

            image_list = list(glob.glob(os.path.join(f"{path_video}/frames", '*.jpg')))
            order_image_list = natsort.natsorted(image_list)
            num_image = len(image_list)

            total_matrix = np.zeros((180, 320))
            for image in range(num_image - 1):
                image1 = order_image_list[image]
                image2 = order_image_list[image + 1]

                first = cv2.imread(image1)
                second = cv2.imread(image2)

                first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
                second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

                first_resize = resize(first_gray, (180, 320))
                second_resize = resize(second_gray, (180, 320))

                array_frame1 = np.array(first_resize)
                array_frame2 = np.array(second_resize)

                check_frames = array_frame1 == array_frame2
                for i in range(len(check_frames)):
                    for j in range(len(check_frames[i])):
                        if not check_frames[i][j]:
                            total_matrix[i][j] += 1

            vector_total_matrix = total_matrix.flatten()
            vector_total_matrix.sort()
            max_value_hm = max(vector_total_matrix / num_image)
            min_value_hm = min(vector_total_matrix / num_image)
            median_hm = statistics.median(vector_total_matrix / num_image)
            mean_hm = statistics.mean(vector_total_matrix / num_image)
            first_quantile_hm = np.quantile(vector_total_matrix / num_image, 0.25)
            third_quantile_hm = np.quantile(vector_total_matrix / num_image, 0.75)

            header = ["id_frame_pair", "frame1", "frame2", "correlation"]
            with open(f"{path_video}/segment-results.csv", 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(header)
                for image in range(num_image - 1):
                    image1 = order_image_list[image]
                    image2 = order_image_list[image + 1]

                    first_image = cv2.imread(image1)
                    second_image = cv2.imread(image2)

                    hsv_image_1 = cv2.cvtColor(first_image, cv2.COLOR_BGR2HSV)
                    hsv_image_2 = cv2.cvtColor(second_image, cv2.COLOR_BGR2HSV)

                    h_bins = 50
                    s_bins = 60
                    histSize = [h_bins, s_bins]
                    h_ranges = [0, 180]
                    s_ranges = [0, 256]
                    ranges = h_ranges + s_ranges
                    channels = [0, 1]

                    hist_image_1 = cv2.calcHist([hsv_image_1], channels, None, histSize, ranges, accumulate=False)
                    cv2.normalize(hist_image_1, hist_image_1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                    hist_image_2 = cv2.calcHist([hsv_image_2], channels, None, histSize, ranges, accumulate=False)
                    cv2.normalize(hist_image_2, hist_image_2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                    compare_method = cv2.HISTCMP_CORREL
                    correlation = cv2.compareHist(hist_image_1, hist_image_2, compare_method)

                    writer.writerow([image, image1, image2, correlation])

            similarity = pd.read_csv(f"{path_video}/segment-results.csv")
            similarity_score = similarity.correlation.abs()

            max_value_similarity = max(similarity_score)
            min_value_similarity = min(similarity_score)
            median_similarity = statistics.median(similarity_score)
            mean_similarity = statistics.mean(similarity_score)
            first_quantile_similarity = np.quantile(similarity_score, 0.25)
            third_quantile_similarity = np.quantile(similarity_score, 0.75)

            data = (max_value_hm, min_value_hm, median_hm, mean_hm, first_quantile_hm, third_quantile_hm,
                    max_value_similarity, min_value_similarity, median_similarity, mean_similarity,
                    first_quantile_similarity, third_quantile_similarity, label)

            sentence = ",".join(map(str, data))
            if sentence:
                arff_file.write(sentence + '\n')
