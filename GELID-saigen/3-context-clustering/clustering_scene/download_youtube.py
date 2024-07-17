import os
import pandas as pd
from chdir import cwd
import ffmpeg
import cv2
import numpy as np
from sys import maxsize
from PIL import Image
import glob
import natsort
import csv
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from skimage.transform import resize
import videokf as vf

def youtubeVideo(path):

    with open(path, 'r') as f: # open in readonly mode
      video_list= pd.read_csv(f)
     
      for idx,row in video_list.iterrows():

            start= row['start-segment(s)']
            matching_keyword = row['matching-word']

            path_video = f"s-{start}-{matching_keyword}"

            segment_file=f"{path_video}/segment.mp4"

        

            vf.extract_keyframes(segment_file)          

            
            image_list = list(glob.glob(os.path.join(f"{path_video}/keyframes",'*.jpg')))
            order_image_list=natsort.natsorted(image_list)
            print(order_image_list)
            num_frame= len(image_list)
            print("\nImages:", num_frame) 

            image_id1=0
            
            total_matrix= np.zeros((180,320,3))
            

            for image in range(num_frame):
                image1 = order_image_list[image_id1]
                
                first= cv2.imread(image1)
                
                #Resize image
                first_resize=resize(first,(180,320,3))

                array_frame = np.array(first_resize)

                total_matrix= total_matrix + array_frame
                    
                image_id1+=1



            median_matrix=np.zeros((180,320,3))  
            new_total_matrix= total_matrix       
            divisor= num_frame
            median_matrix= np.divide(new_total_matrix, divisor)

            fig= plt.imshow(median_matrix, vmin=0, vmax=100)

            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                        hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            aa=plt.savefig(f"final_frame_{idx}.jpg", bbox_inches = 'tight', pad_inches = 0)
            

    ######SIMILARITY######

    image_list = list(glob.glob(os.path.join("",'*.jpg')))
    image_list_compare = list(glob.glob(os.path.join("",'*.jpg')))

    order_image_list=natsort.natsorted(image_list)
    print(order_image_list)
    num_image= len(image_list)
    print("\nImages:", num_image) 


    id_frame_pair=-1

    list_name_feature=[]
    for i in range(num_image):
        f=f"f_{i}"
        list_name_feature.append(f)


    header=list_name_feature


    with open('similarity.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for element in range(len(image_list)):
            features_similarity=[]
            for compare in range(len(image_list_compare)):
                image_1 = order_image_list[element]
                image_2 = order_image_list[compare]


                first_image= cv2.imread(image_1)
                second_image= cv2.imread(image_2)

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

                output_score=1-correlation
                features_similarity.append(output_score)

            writer.writerow(features_similarity)

    csvfile.close()

     

                                

                

                

            
    
    
