# from ffmpy import FFmpeg
# import ffmpy
# import ffmpeg
# import json
# import os 
# import spacy
# import numpy as np
# import pandas as pd
# from chdir import cwd
# # import traceback
# # import sys

# def videoTrimTrashold(file, keyword, threshold):
 
#     nlp = spacy.load('en_core_web_sm')

#     os.makedirs("segments", exist_ok=True)
#     print("Threshold to trim video") 

#     print("threshold: ", threshold)

#     with open(file) as f:
#         # returns JSON object as 
#         # a dictionary
#         data = json.load(f)
        
#         last_element=data[-1]
#         end_video=int(last_element['start'])+int(last_element['duration'])
#         # Iterating through the json
#         # list
#         for element in data:
#             caption= str(element['text'])
#             start_caption= int(element['start'])
#             duration_caption= int(element['duration'])
#             end_caption= start_caption + duration_caption
#             words= nlp(caption.lower())


#             words_caption= [token.text for token in words]

#             for w in words_caption:
#                 for k in keyword:
#                     if w == k:
#                         time_start= start_caption-int(threshold)
#                         time_end= end_caption + int(threshold)

#                         if time_end > end_video:
#                             time_end=end_video

#                         in_file = ffmpeg.input('video.mp4')  #####CHECKTOPASSVIDEO###
#                         segment_folder = f"segments/s-{time_start}-{k}"
                        
#                         try:
#                             os.makedirs(segment_folder, exist_ok=True)
#                             (
#                                 in_file.trim(start=time_start, end=time_end).setpts('PTS-STARTPTS')
#                                 .crop(190, 10, 450, 320)
#                                 .output(f"{segment_folder}/segment.mp4")
#                                 .run(overwrite_output=True)
#                                 )
#                         except :
#                             # etype, value, tb = sys.exc_info()
    
#                             # traceback.format_exception(etype, value, tb) # list[str] を返す
#                             # traceback.format_tb(tb) # list[str] を返す
#                             # traceback.format_exception_only(etype, value) # list[str] を返す
    
#                             # traceback.format_exc() # tuple[str] を返す
#                             # print(traceback.format_exception(etype, value, tb))
#                             print("segment analysed")
                    
#                     # else:
#                     #     print("Non-matching keyword")

from ffmpy import FFmpeg
import ffmpy
import ffmpeg
import json
import os 
import spacy
import numpy as np
import pandas as pd
from chdir import cwd
import csv  # CSVモジュールのインポート

def videoTrimTrashold(file, keyword, threshold):
 
    nlp = spacy.load('en_core_web_sm')

    os.makedirs("segments", exist_ok=True)
    print("Threshold to trim video") 
    print("threshold: ", threshold)

    csv_file = 'segments/segments_info.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['time_start', 'time_end', 'caption'])  # ヘッダーを書き込む

        with open(file) as f:
            # JSONオブジェクトを辞書として返す
            data = json.load(f)
            
            last_element = data[-1]
            end_video = int(last_element['start']) + int(last_element['duration'])

            # JSONリストを反復処理
            for element in data:
                caption = str(element['text'])
                start_caption = int(element['start'])
                duration_caption = int(element['duration'])
                end_caption = start_caption + duration_caption
                words = nlp(caption.lower())

                words_caption = [token.text for token in words]

                for w in words_caption:
                    for k in keyword:
                        if w == k:
                            time_start = start_caption - int(threshold)
                            time_end = end_caption + int(threshold)

                            if time_end > end_video:
                                time_end = end_video

                            # CSVにデータを書き込む
                            writer.writerow([time_start, time_end, caption])
                            print(f"Matching keyword '{k}' found. Written to CSV: start={time_start}, end={time_end}, caption={caption}")

                            in_file = ffmpeg.input('video.mp4')  #####CHECKTOPASSVIDEO###
                            segment_folder = f"segments/s-{time_start}-{k}"
                            
                            try:
                                os.makedirs(segment_folder, exist_ok=True)
                                (
                                    in_file.trim(start=time_start, end=time_end).setpts('PTS-STARTPTS')
                                    .crop(190, 10, 450, 320)
                                    .output(f"{segment_folder}/segment.mp4")
                                    .run(overwrite_output=True)
                                )
                            except:
                                print("segment analysed")
