# import os
# import nltk
# import spacy
# import pandas as pd
# from pytube import YouTube
# from chdir import cwd
# import ffmpeg
# import cv2
# import cv2 as cv
# import numpy as np
# from sys import maxsize
# from PIL import Image
# import glob
# import natsort
# import csv
# import pandas as pd
# import statistics
# from skimage.transform import resize
# from gensim.models import Word2Vec, KeyedVectors
# from spacy.tokens import token

# model_path=os.path.abspath("GoogleNews-vectors-negative300.bin")
# model = KeyedVectors.load_word2vec_format(model_path, binary=True)



# def youtubeVideo(path):
#   nlp = spacy.load('en_core_web_sm')
#   stopwords = spacy.lang.en.stop_words.STOP_WORDS

#   list_name_feature = [f"f_{i}" for i in range(313)]
#   header = list_name_feature

#   with open("feature_cluster_issue.csv", "w", newline='') as cluster_file:
#     cluster_writer = csv.writer(cluster_file)
#     cluster_writer.writerow(header)

#     with open(path, 'r') as f: # open in readonly mode
#         video_list= pd.read_csv(f)

#         for idx,row in video_list.iterrows():
#             subtitle=row['caption']
#             try:
#                 sub=subtitle.lower()
#             except:
#                 print("no-lower-caption")    
#             doc = nlp(sub)

#             tokens = [token.text for token in doc]
            

#             # Generate lemmatized tokens
#             lemmas = [token.lemma_ for token in doc]

#             # Remove stopwords and non-alphabetic tokens
#             a_lemmas = [lemma for lemma in lemmas if lemma.isalpha() and lemma in model and lemma not in stopwords]
#             print(a_lemmas)

#             vectors = [model.get_vector(lemma) for lemma in a_lemmas]
#             if (len(vectors) == 0):
#                 continue

#             result = []
#             for i in range(0, len(vectors[0])):
#                 sum = 0.0
#                 for j in range(0, len(vectors)):
#                     sum += vectors[j][i]
#                 average = sum / len(vectors)
#                 result.append(average)
            
#             print(len(result))

#             if len(result) == 300:  # 確実に300個の数値がある場合
#                 cluster_writer.writerow(result + [0] * (313 - 300))  # 300個以上の場合は0で埋める
#             else:
#                 cluster_writer.writerow(result + [0] * (313 - len(result)))  # 313個以上になるように0で埋める
  
# #   list_name_feature=[]
# #   for i in range(313):
# #     f=f"f_{i}"
# #     list_name_feature.append(f)


# #   header=list_name_feature
  
# #   with open("feature_cluster_issue.csv", "w", newline='') as cluster_file:
# #     cluster_writer = csv.writer(cluster_file)
# #     cluster_writer.writerow(header)

# #     nlp = spacy.load('en_core_web_sm')
# #     stopwords = spacy.lang.en.stop_words.STOP_WORDS

# #     with open(path, 'r') as f: # open in readonly mode
# #         video_list= pd.read_csv(f)

# #         for idx,row in video_list.iterrows():
# #             subtitle=row['caption']
# #             try:
# #                 sub=subtitle.lower()
# #             except:
# #                 print("no-lower-caption")    
# #             doc = nlp(sub)

# #             # Generate the tokens
# #             tokens = [token.text for token in doc]
            

# #             # Generate lemmatized tokens
# #             lemmas = [token.lemma_ for token in doc]

# #             # Remove stopwords and non-alphabetic tokens
# #             a_lemmas = [lemma for lemma in lemmas if lemma.isalpha() and lemma in model and lemma not in stopwords]
# #             print(a_lemmas)

# #             vectors = [model.get_vector(lemma) for lemma in a_lemmas]
# #             if (len(vectors) == 0):
# #                 continue

# #             result = []
# #             for i in range(0, len(vectors[0])):
# #                 sum = 0.0
# #                 for j in range(0, len(vectors)):
# #                     sum += vectors[j][i]
# #                 average = sum / len(vectors)
# #                 result.append(average)
            
# #             print(len(result))

# #             features_list = list(result)
# #             features_w2v=','.join(str(e) for e in features_list)

#             start= row['start-segment(s)']
#             matching_keyword = row['matching-word']

#             path_video = f"s-{start}-{matching_keyword}"

#             segment_file=f"{path_video}/segment.mp4"
#             print("Start frame extraction")
#             os.mkdir(f"{path_video}/frames")
            
#             video_capture = cv2.VideoCapture(segment_file)
#             success,image = video_capture.read()
#             count = 0
#             while success:
#                 cv2.imwrite(f"{path_video}/frames/%d.jpg" % count, image)  
#                 success,image = video_capture.read()
#                 print('Read a new frame: ', success)
#                 count += 1

#             #****************************#
#             image_list = list(glob.glob(os.path.join(f"{path_video}/frames",'*.jpg')))
#             order_image_list=natsort.natsorted(image_list)
#             print(order_image_list)
#             num_image= len(image_list)
#             print("\nImages:", num_image) 

#             image_id1=0
#             image_id2=1

#             total_matrix= np.zeros((180,320))


#             for image in range(num_image-1):
#                 image1 = order_image_list[image_id1]
#                 image2 = order_image_list[image_id2]
                
#                 first= cv.imread(image1)
#                 second= cv.imread(image2)

#                 # Convert images to grayscale
#                 first_gray = cv.cvtColor(first, cv.COLOR_BGR2GRAY)
#                 second_gray = cv.cvtColor(second, cv.COLOR_BGR2GRAY)

#                 #Resize image
#                 first_resize=resize(first_gray,(180,320))
#                 second_resize=resize(second_gray,(180,320))

#                 array_frame1 = np.array(first_resize)
#                 array_frame2 = np.array(second_resize)    

#                 check_frames = array_frame1 == array_frame2

#                 for i in range(0,len(check_frames)):
#                     for j in range(len(check_frames[i])):
#                         if check_frames[i][j] == False:
#                             total_matrix[i][j] = total_matrix[i][j]+1
                    
#                 image_id1+=1
#                 image_id2+=1


        
#             vector_total_matrix= total_matrix.flatten()
            
#             vector_total_matrix.sort()
#             print('List in Ascending Order: ', vector_total_matrix)
#             vector_len=len(vector_total_matrix)
#             max_value_hm = max(vector_total_matrix/num_image)
#             print('Maximum value:', max_value_hm)
#             min_value_hm = min(vector_total_matrix/num_image)
#             print('Min value:', min_value_hm)
            
#             median_hm = statistics.median(vector_total_matrix/num_image)
#             print('Median', median_hm)
#             mean_hm = statistics.mean(vector_total_matrix/num_image)
#             print('Mean', mean_hm)
            
#             first_quantile_hm=np.quantile(vector_total_matrix/num_image, 0.25)
#             print('1Q', first_quantile_hm)
#             third_quantile_hm=np.quantile(vector_total_matrix/num_image, 0.75)
#             print('3Q', third_quantile_hm)


#             #****************************#
#             image_list = list(glob.glob(os.path.join(f"{path_video}/frames",'*.jpg')))
#             order_image_list=natsort.natsorted(image_list)

#             num_image= len(image_list)
#             print("\nImages:", num_image)
#             count=num_image/2

#             image_id_1=0
#             image_id_2=1
#             id_frame_pair=-1

#             header=["id_frame_pair","frame1", "frame2", "correlation"]

#             with open(f"{path_video}/segment-results.csv", 'w', newline='') as csv_file:
#                 writer = csv.writer(csv_file)
#                 writer.writerow(header)
#                 for image in range(num_image-1):
#                     image_1 = order_image_list[image_id_1]
#                     image_2 = order_image_list[image_id_2]


#                     first_image= cv.imread(image_1)
#                     second_image= cv.imread(image_2)

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

#                     frame_1= order_image_list[image_id_1]
#                     frame_2=order_image_list[image_id_2]

#                     frame_pair1=frame_1
#                     frame_pair2=frame_2

#                     image_id_1+=1
#                     image_id_2+=1
#                     id_frame_pair+=1

#                     writer.writerow([id_frame_pair, frame_pair1, frame_pair2, correlation])


#             csv_file.close

#             similarity = pd.read_csv(f"{path_video}/segment-results.csv")
#             similarity_score=similarity.correlation

        
#             abs_similarity_score=abs(similarity_score)
#             max_value_similarity = max(abs_similarity_score)
#             print('Maximum value:', max_value_similarity)
#             min_value_similarity = min(abs_similarity_score)
#             print('Min value:', min_value_similarity)
            
#             median_similarity = statistics.median(abs_similarity_score)
#             print('Median', median_similarity)
#             mean_similarity = statistics.mean(abs_similarity_score)
#             print('Mean', mean_similarity)

#             first_quantile_similarity=np.quantile(abs_similarity_score, 0.25)
#             print("1Q: ", first_quantile_similarity)
#             third_quantile_similarity=np.quantile(abs_similarity_score, 0.75)
#             print("3Q: ", third_quantile_similarity)
                
#             print(features_w2v)

#             cluster_writer.writerow([features_w2v,max_value_hm,min_value_hm,median_hm,mean_hm,first_quantile_hm,third_quantile_hm,max_value_similarity, min_value_similarity, median_similarity,mean_similarity,mean_similarity,first_quantile_similarity,third_quantile_similarity])
        
#   cluster_file.close

import os
import nltk
import spacy
import pandas as pd
from pytube import YouTube
from chdir import cwd
import ffmpeg
import cv2
import numpy as np
from PIL import Image
import glob
import natsort
import csv
import statistics
from skimage.transform import resize
from gensim.models import Word2Vec, KeyedVectors

model_path = os.path.abspath("GoogleNews-vectors-negative300.bin")
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

def youtubeVideo(path):
    nlp = spacy.load('en_core_web_sm')
    stopwords = spacy.lang.en.stop_words.STOP_WORDS

    list_name_feature = [f"f_{i}" for i in range(312)]
    header = list_name_feature

    with open("uenaka_feature_cluster_issue.csv", "w", newline='') as cluster_file:
        cluster_writer = csv.writer(cluster_file)
        cluster_writer.writerow(header)

        with open(path, 'r') as f:  # open in readonly mode
            video_list = pd.read_csv(f)

            for idx, row in video_list.iterrows():
                subtitle = row['caption']
                try:
                    sub = subtitle.lower()
                except:
                    print("no-lower-caption")
                    continue

                doc = nlp(sub)
                lemmas = [token.lemma_ for token in doc if token.lemma_.isalpha() and token.lemma_ in model and token.lemma_ not in stopwords]
                vectors = [model.get_vector(lemma) for lemma in lemmas]
                if not vectors:
                    continue

                result = np.mean(vectors, axis=0).tolist()
                if len(result) < 300:
                    result.extend([0] * (300 - len(result)))

                start = row['start-segment(s)']
                matching_keyword = row['matching-word']
                path_video = f"s-{start}-{matching_keyword}"

                segment_file = f"{path_video}/segment.mp4"
                print("Start frame extraction")
                os.makedirs(f"{path_video}/frames", exist_ok=True)

                video_capture = cv2.VideoCapture(segment_file)
                success, image = video_capture.read()
                count = 0
                while success:
                    cv2.imwrite(f"{path_video}/frames/%d.jpg" % count, image)
                    success, image = video_capture.read()
                    count += 1

                image_list = natsort.natsorted(glob.glob(os.path.join(f"{path_video}/frames", '*.jpg')))
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
                with open(f"{path_video}/segment-results.csv", 'w', newline='') as csv_file:
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

                similarity = pd.read_csv(f"{path_video}/segment-results.csv")
                abs_similarity_score = similarity['correlation'].abs()
                
                max_value_similarity = abs_similarity_score.max()
                min_value_similarity = abs_similarity_score.min()
                median_similarity = abs_similarity_score.median()
                mean_similarity = abs_similarity_score.mean()
                first_quantile_similarity = abs_similarity_score.quantile(0.25)
                third_quantile_similarity = abs_similarity_score.quantile(0.75)

                result.extend([
                    max_value_hm, min_value_hm, median_hm, mean_hm, first_quantile_hm, third_quantile_hm,
                    max_value_similarity, min_value_similarity, median_similarity, mean_similarity,
                    first_quantile_similarity, third_quantile_similarity
                ])

                cluster_writer.writerow(result)

    cluster_file.close()
