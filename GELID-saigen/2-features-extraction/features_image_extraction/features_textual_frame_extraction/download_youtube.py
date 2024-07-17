import os
import nltk
import spacy
import pandas as pd
from pytube import YouTube
from chdir import cwd
import ffmpeg
import cv2
import cv2 as cv
import numpy as np
from sys import maxsize
from PIL import Image
import glob
import natsort
import csv
import pandas as pd
import statistics
from skimage.transform import resize
from gensim.models import Word2Vec, KeyedVectors
from spacy.tokens import token
# import traceback
# import sys

model_path=os.path.abspath("GoogleNews-vectors-negative300.bin")
model = KeyedVectors.load_word2vec_format(model_path, binary=True)



SAVE_PATH = "video"

def youtubeVideo(path):
  with open("frame_extraction.arff", "w") as arff_file:
    arff_file.write('''@RELATION classification
@ATTRIBUTE f1 REAL
@ATTRIBUTE f2 REAL
@ATTRIBUTE f3 REAL
@ATTRIBUTE f4 REAL
@ATTRIBUTE f5 REAL
@ATTRIBUTE f6 REAL
@ATTRIBUTE f7 REAL
@ATTRIBUTE f8 REAL
@ATTRIBUTE f9 REAL
@ATTRIBUTE f10 REAL
@ATTRIBUTE f11 REAL
@ATTRIBUTE f12 REAL
@ATTRIBUTE f13 REAL
@ATTRIBUTE f14 REAL
@ATTRIBUTE f15 REAL
@ATTRIBUTE f16 REAL
@ATTRIBUTE f17 REAL
@ATTRIBUTE f18 REAL
@ATTRIBUTE f19 REAL
@ATTRIBUTE f20 REAL
@ATTRIBUTE f21 REAL
@ATTRIBUTE f22 REAL
@ATTRIBUTE f23 REAL
@ATTRIBUTE f24 REAL
@ATTRIBUTE f25 REAL
@ATTRIBUTE f26 REAL
@ATTRIBUTE f27 REAL
@ATTRIBUTE f28 REAL
@ATTRIBUTE f29 REAL
@ATTRIBUTE f30 REAL
@ATTRIBUTE f31 REAL
@ATTRIBUTE f32 REAL
@ATTRIBUTE f33 REAL
@ATTRIBUTE f34 REAL
@ATTRIBUTE f35 REAL
@ATTRIBUTE f36 REAL
@ATTRIBUTE f37 REAL
@ATTRIBUTE f38 REAL
@ATTRIBUTE f39 REAL
@ATTRIBUTE f40 REAL
@ATTRIBUTE f41 REAL
@ATTRIBUTE f42 REAL
@ATTRIBUTE f43 REAL
@ATTRIBUTE f44 REAL
@ATTRIBUTE f45 REAL
@ATTRIBUTE f46 REAL
@ATTRIBUTE f47 REAL
@ATTRIBUTE f48 REAL
@ATTRIBUTE f49 REAL
@ATTRIBUTE f50 REAL
@ATTRIBUTE f51 REAL
@ATTRIBUTE f52 REAL
@ATTRIBUTE f53 REAL
@ATTRIBUTE f54 REAL
@ATTRIBUTE f55 REAL
@ATTRIBUTE f56 REAL
@ATTRIBUTE f57 REAL
@ATTRIBUTE f58 REAL
@ATTRIBUTE f59 REAL
@ATTRIBUTE f60 REAL
@ATTRIBUTE f61 REAL
@ATTRIBUTE f62 REAL
@ATTRIBUTE f63 REAL
@ATTRIBUTE f64 REAL
@ATTRIBUTE f65 REAL
@ATTRIBUTE f66 REAL
@ATTRIBUTE f67 REAL
@ATTRIBUTE f68 REAL
@ATTRIBUTE f69 REAL
@ATTRIBUTE f70 REAL
@ATTRIBUTE f71 REAL
@ATTRIBUTE f72 REAL
@ATTRIBUTE f73 REAL
@ATTRIBUTE f74 REAL
@ATTRIBUTE f75 REAL
@ATTRIBUTE f76 REAL
@ATTRIBUTE f77 REAL
@ATTRIBUTE f78 REAL
@ATTRIBUTE f79 REAL
@ATTRIBUTE f80 REAL
@ATTRIBUTE f81 REAL
@ATTRIBUTE f82 REAL
@ATTRIBUTE f83 REAL
@ATTRIBUTE f84 REAL
@ATTRIBUTE f85 REAL
@ATTRIBUTE f86 REAL
@ATTRIBUTE f87 REAL
@ATTRIBUTE f88 REAL
@ATTRIBUTE f89 REAL
@ATTRIBUTE f90 REAL
@ATTRIBUTE f91 REAL
@ATTRIBUTE f92 REAL
@ATTRIBUTE f93 REAL
@ATTRIBUTE f94 REAL
@ATTRIBUTE f95 REAL
@ATTRIBUTE f96 REAL
@ATTRIBUTE f97 REAL
@ATTRIBUTE f98 REAL
@ATTRIBUTE f99 REAL
@ATTRIBUTE f100 REAL
@ATTRIBUTE f101 REAL
@ATTRIBUTE f102 REAL
@ATTRIBUTE f103 REAL
@ATTRIBUTE f104 REAL
@ATTRIBUTE f105 REAL
@ATTRIBUTE f106 REAL
@ATTRIBUTE f107 REAL
@ATTRIBUTE f108 REAL
@ATTRIBUTE f109 REAL
@ATTRIBUTE f110 REAL
@ATTRIBUTE f111 REAL
@ATTRIBUTE f112 REAL
@ATTRIBUTE f113 REAL
@ATTRIBUTE f114 REAL
@ATTRIBUTE f115 REAL
@ATTRIBUTE f116 REAL
@ATTRIBUTE f117 REAL
@ATTRIBUTE f118 REAL
@ATTRIBUTE f119 REAL
@ATTRIBUTE f120 REAL
@ATTRIBUTE f121 REAL
@ATTRIBUTE f122 REAL
@ATTRIBUTE f123 REAL
@ATTRIBUTE f124 REAL
@ATTRIBUTE f125 REAL
@ATTRIBUTE f126 REAL
@ATTRIBUTE f127 REAL
@ATTRIBUTE f128 REAL
@ATTRIBUTE f129 REAL
@ATTRIBUTE f130 REAL
@ATTRIBUTE f131 REAL
@ATTRIBUTE f132 REAL
@ATTRIBUTE f133 REAL
@ATTRIBUTE f134 REAL
@ATTRIBUTE f135 REAL
@ATTRIBUTE f136 REAL
@ATTRIBUTE f137 REAL
@ATTRIBUTE f138 REAL
@ATTRIBUTE f139 REAL
@ATTRIBUTE f140 REAL
@ATTRIBUTE f141 REAL
@ATTRIBUTE f142 REAL
@ATTRIBUTE f143 REAL
@ATTRIBUTE f144 REAL
@ATTRIBUTE f145 REAL
@ATTRIBUTE f146 REAL
@ATTRIBUTE f147 REAL
@ATTRIBUTE f148 REAL
@ATTRIBUTE f149 REAL
@ATTRIBUTE f150 REAL
@ATTRIBUTE f151 REAL
@ATTRIBUTE f152 REAL
@ATTRIBUTE f153 REAL
@ATTRIBUTE f154 REAL
@ATTRIBUTE f155 REAL
@ATTRIBUTE f156 REAL
@ATTRIBUTE f157 REAL
@ATTRIBUTE f158 REAL
@ATTRIBUTE f159 REAL
@ATTRIBUTE f160 REAL
@ATTRIBUTE f161 REAL
@ATTRIBUTE f162 REAL
@ATTRIBUTE f163 REAL
@ATTRIBUTE f164 REAL
@ATTRIBUTE f165 REAL
@ATTRIBUTE f166 REAL
@ATTRIBUTE f167 REAL
@ATTRIBUTE f168 REAL
@ATTRIBUTE f169 REAL
@ATTRIBUTE f170 REAL
@ATTRIBUTE f171 REAL
@ATTRIBUTE f172 REAL
@ATTRIBUTE f173 REAL
@ATTRIBUTE f174 REAL
@ATTRIBUTE f175 REAL
@ATTRIBUTE f176 REAL
@ATTRIBUTE f177 REAL
@ATTRIBUTE f178 REAL
@ATTRIBUTE f179 REAL
@ATTRIBUTE f180 REAL
@ATTRIBUTE f181 REAL
@ATTRIBUTE f182 REAL
@ATTRIBUTE f183 REAL
@ATTRIBUTE f184 REAL
@ATTRIBUTE f185 REAL
@ATTRIBUTE f186 REAL
@ATTRIBUTE f187 REAL
@ATTRIBUTE f188 REAL
@ATTRIBUTE f189 REAL
@ATTRIBUTE f190 REAL
@ATTRIBUTE f191 REAL
@ATTRIBUTE f192 REAL
@ATTRIBUTE f193 REAL
@ATTRIBUTE f194 REAL
@ATTRIBUTE f195 REAL
@ATTRIBUTE f196 REAL
@ATTRIBUTE f197 REAL
@ATTRIBUTE f198 REAL
@ATTRIBUTE f199 REAL
@ATTRIBUTE f200 REAL
@ATTRIBUTE f201 REAL
@ATTRIBUTE f202 REAL
@ATTRIBUTE f203 REAL
@ATTRIBUTE f204 REAL
@ATTRIBUTE f205 REAL
@ATTRIBUTE f206 REAL
@ATTRIBUTE f207 REAL
@ATTRIBUTE f208 REAL
@ATTRIBUTE f209 REAL
@ATTRIBUTE f210 REAL
@ATTRIBUTE f211 REAL
@ATTRIBUTE f212 REAL
@ATTRIBUTE f213 REAL
@ATTRIBUTE f214 REAL
@ATTRIBUTE f215 REAL
@ATTRIBUTE f216 REAL
@ATTRIBUTE f217 REAL
@ATTRIBUTE f218 REAL
@ATTRIBUTE f219 REAL
@ATTRIBUTE f220 REAL
@ATTRIBUTE f221 REAL
@ATTRIBUTE f222 REAL
@ATTRIBUTE f223 REAL
@ATTRIBUTE f224 REAL
@ATTRIBUTE f225 REAL
@ATTRIBUTE f226 REAL
@ATTRIBUTE f227 REAL
@ATTRIBUTE f228 REAL
@ATTRIBUTE f229 REAL
@ATTRIBUTE f230 REAL
@ATTRIBUTE f231 REAL
@ATTRIBUTE f232 REAL
@ATTRIBUTE f233 REAL
@ATTRIBUTE f234 REAL
@ATTRIBUTE f235 REAL
@ATTRIBUTE f236 REAL
@ATTRIBUTE f237 REAL
@ATTRIBUTE f238 REAL
@ATTRIBUTE f239 REAL
@ATTRIBUTE f240 REAL
@ATTRIBUTE f241 REAL
@ATTRIBUTE f242 REAL
@ATTRIBUTE f243 REAL
@ATTRIBUTE f244 REAL
@ATTRIBUTE f245 REAL
@ATTRIBUTE f246 REAL
@ATTRIBUTE f247 REAL
@ATTRIBUTE f248 REAL
@ATTRIBUTE f249 REAL
@ATTRIBUTE f250 REAL
@ATTRIBUTE f251 REAL
@ATTRIBUTE f252 REAL
@ATTRIBUTE f253 REAL
@ATTRIBUTE f254 REAL
@ATTRIBUTE f255 REAL
@ATTRIBUTE f256 REAL
@ATTRIBUTE f257 REAL
@ATTRIBUTE f258 REAL
@ATTRIBUTE f259 REAL
@ATTRIBUTE f260 REAL
@ATTRIBUTE f261 REAL
@ATTRIBUTE f262 REAL
@ATTRIBUTE f263 REAL
@ATTRIBUTE f264 REAL
@ATTRIBUTE f265 REAL
@ATTRIBUTE f266 REAL
@ATTRIBUTE f267 REAL
@ATTRIBUTE f268 REAL
@ATTRIBUTE f269 REAL
@ATTRIBUTE f270 REAL
@ATTRIBUTE f271 REAL
@ATTRIBUTE f272 REAL
@ATTRIBUTE f273 REAL
@ATTRIBUTE f274 REAL
@ATTRIBUTE f275 REAL
@ATTRIBUTE f276 REAL
@ATTRIBUTE f277 REAL
@ATTRIBUTE f278 REAL
@ATTRIBUTE f279 REAL
@ATTRIBUTE f280 REAL
@ATTRIBUTE f281 REAL
@ATTRIBUTE f282 REAL
@ATTRIBUTE f283 REAL
@ATTRIBUTE f284 REAL
@ATTRIBUTE f285 REAL
@ATTRIBUTE f286 REAL
@ATTRIBUTE f287 REAL
@ATTRIBUTE f288 REAL
@ATTRIBUTE f289 REAL
@ATTRIBUTE f290 REAL
@ATTRIBUTE f291 REAL
@ATTRIBUTE f292 REAL
@ATTRIBUTE f293 REAL
@ATTRIBUTE f294 REAL
@ATTRIBUTE f295 REAL
@ATTRIBUTE f296 REAL
@ATTRIBUTE f297 REAL
@ATTRIBUTE f298 REAL
@ATTRIBUTE f299 REAL
@ATTRIBUTE f300 REAL    
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
@ATTRIBUTE classification {non-informative, presentation, logic, balance, performance}

@DATA 
''')
    nlp = spacy.load('en_core_web_sm')
    stopwords = spacy.lang.en.stop_words.STOP_WORDS

    with open(path, 'r') as f: # open in readonly mode
        video_list= pd.read_csv(f)
     
        for idx,row in video_list.iterrows():
            subtitle=row['caption']
            try:
                sub=subtitle.lower()
            except:
                sub=subtitle
            doc = nlp(sub)

            # Generate the tokens
            tokens = [token.text for token in doc]
            

            # Generate lemmatized tokens
            lemmas = [token.lemma_ for token in doc]

            # Remove stopwords and non-alphabetic tokens
            a_lemmas = [lemma for lemma in lemmas if lemma.isalpha() and lemma in model and lemma not in stopwords]
            print(a_lemmas)

            #tokens = [token.text for token in doc if token.text in model and token.text not in stopwords]
            #print(tokens)

            vectors = [model.get_vector(lemma) for lemma in a_lemmas]
            if (len(vectors) == 0):
                continue

            result = []
            for i in range(0, len(vectors[0])):
                sum = 0.0
                for j in range(0, len(vectors)):
                    sum += vectors[j][i]
                average = sum / len(vectors)
                result.append(average)
            
            print(len(result))

            features_list = list(result)
            features_w2v=','.join(str(e) for e in features_list)

            label=row['labeled']
            start= row['start-segment(s)']
            matching_keyword = row['matching-word']

            path_video = f"s-{start}-{matching_keyword}"

            os.mkdir(f"{path_video}/frames")
            segment_file = f"{path_video}/segment.mp4"
            
            video_capture = cv2.VideoCapture(segment_file)
            success,image = video_capture.read()
            count = 0
            while success:
                cv2.imwrite(f"{path_video}/frames/%d.jpg" % count, image)  
                success,image = video_capture.read()
                print('Read a new frame: ', success)
                count += 1

            #****************************#
            image_list = list(glob.glob(os.path.join(f"{path_video}/frames",'*.jpg')))
            order_image_list=natsort.natsorted(image_list)
            print(order_image_list)
            num_image= len(image_list)
            print("\nImages:", num_image) 

            image_id1=0
            image_id2=1

            total_matrix= np.zeros((180,320))


            for image in range(num_image-1):
                image1 = order_image_list[image_id1]
                image2 = order_image_list[image_id2]
                
                first= cv.imread(image1)
                second= cv.imread(image2)

                # Convert images to grayscale
                first_gray = cv.cvtColor(first, cv.COLOR_BGR2GRAY)
                second_gray = cv.cvtColor(second, cv.COLOR_BGR2GRAY)

                #Resize image
                first_resize=resize(first_gray,(180,320))
                second_resize=resize(second_gray,(180,320))

                array_frame1 = np.array(first_resize)
                array_frame2 = np.array(second_resize)    

                check_frames = array_frame1 == array_frame2
                #print(check_frames)

                for i in range(0,len(check_frames)):
                    for j in range(len(check_frames[i])):
                        if check_frames[i][j] == False:
                            total_matrix[i][j] = total_matrix[i][j]+1
                    
                image_id1+=1
                image_id2+=1


            #####ANALYSIS HEATMAP########
            #SCRIVERE I RISULTATI IN UN FILE #per ogni matrice un vettore
            vector_total_matrix= total_matrix.flatten()
            #print(total_matrix.flatten())
            #ordino il vettore
            vector_total_matrix.sort()
            print('List in Ascending Order: ', vector_total_matrix)
            vector_len=len(vector_total_matrix)
            max_value_hm = max(vector_total_matrix/num_image)
            print('Maximum value:', max_value_hm)
            min_value_hm = min(vector_total_matrix/num_image)
            print('Min value:', min_value_hm)
            
            median_hm = statistics.median(vector_total_matrix/num_image)
            print('Median', median_hm)
            mean_hm = statistics.mean(vector_total_matrix/num_image)
            print('Mean', mean_hm)
            
            first_quantile_hm=np.quantile(vector_total_matrix/num_image, 0.25)
            print('1Q', first_quantile_hm)
            third_quantile_hm=np.quantile(vector_total_matrix/num_image, 0.75)
            print('3Q', third_quantile_hm)


            #****************************#
            image_list = list(glob.glob(os.path.join(f"{path_video}/frames",'*.jpg')))
            order_image_list=natsort.natsorted(image_list)

            num_image= len(image_list)
            print("\nImages:", num_image)
            count=num_image/2

            image_id_1=0
            image_id_2=1
            id_frame_pair=-1

            header=["id_frame_pair","frame1", "frame2", "correlation"]

            with open(f"{path_video}/segment-results.csv", 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(header)
                for image in range(num_image-1):
                    image_1 = order_image_list[image_id_1]
                    image_2 = order_image_list[image_id_2]


                    first_image= cv.imread(image_1)
                    second_image= cv.imread(image_2)

                    hsv_image_1 = cv.cvtColor(first_image, cv.COLOR_BGR2HSV)
                    hsv_image_2 = cv.cvtColor(second_image, cv.COLOR_BGR2HSV)

                    h_bins = 50
                    s_bins = 60
                    histSize = [h_bins, s_bins]
                    h_ranges = [0, 180]
                    s_ranges = [0, 256]
                    ranges = h_ranges + s_ranges
                    channels = [0, 1]

                    hist_image_1 = cv.calcHist([hsv_image_1], channels, None, histSize, ranges, accumulate=False)
                    cv.normalize(hist_image_1, hist_image_1, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

                    hist_image_2 = cv.calcHist([hsv_image_2], channels, None, histSize, ranges, accumulate=False)
                    cv.normalize(hist_image_2, hist_image_2, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)


                    compare_method = cv.HISTCMP_CORREL

                    #score_image1 = cv.compareHist(hist_image_1, hist_image_1, compare_method)
                    correlation = cv.compareHist(hist_image_1, hist_image_2, compare_method)

                    frame_1= order_image_list[image_id_1]
                    frame_2=order_image_list[image_id_2]

                    frame_pair1=frame_1
                    frame_pair2=frame_2

                    image_id_1+=1
                    image_id_2+=1
                    id_frame_pair+=1

                    writer.writerow([id_frame_pair, frame_pair1, frame_pair2, correlation])


            csv_file.close

            similarity = pd.read_csv(f"{path_video}/segment-results.csv")
            similarity_score=similarity.correlation

            try:

                abs_similarity_score=abs(similarity_score)
                max_value_similarity = max(abs_similarity_score)
                print('Maximum value:', max_value_similarity)
                min_value_similarity = min(abs_similarity_score)
                print('Min value:', min_value_similarity)
                
                median_similarity = statistics.median(abs_similarity_score)
                print('Median', median_similarity)
                mean_similarity = statistics.mean(abs_similarity_score)
                print('Mean', mean_similarity)

                first_quantile_similarity=np.quantile(abs_similarity_score, 0.25)
                print("1Q: ", first_quantile_similarity)
                third_quantile_similarity=np.quantile(abs_similarity_score, 0.75)
                print("3Q: ", third_quantile_similarity)
                    

                data=features_w2v, max_value_hm, min_value_hm, median_hm, mean_hm, first_quantile_hm, third_quantile_hm, max_value_similarity, min_value_similarity, median_similarity, mean_similarity, first_quantile_similarity, third_quantile_similarity, label
                sent_new = data
                print(sent_new)
                sentence="".join(str(sent_new)).replace("(","").replace(")","").replace("'","")
                if sentence != "":
                    arff_file.write(sentence)
                    arff_file.write("\n")   
            
            except:
                print("no info")
