import json
import os
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from chdir import cwd
import csv

SAVE_PATH = "video"

def youtubeVideo(path):
    
    #create directory for each video
    # print(f"path = {path}")
    print(os.getcwd())
    with open(path, 'r') as link_file:
        lines = link_file.readlines()
        print(lines)
        for i in range(len(lines)):
            path_video = f"v{i}"
            if not os.path.exists(path_video):
                os.mkdir(path_video)
            
            line = lines[i] 
            try: 
                my_video = YouTube(line) 
            except: 
                print("Connection Error")
            
            with cwd(path_video):

                header=["youtube_title", "youtube_description", "length", "is_bug_video"]

                with open('youtube_video_info.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(header)

                    print("********************Download INFO*************************")
                    try:
                        title= my_video.title
                    except:
                        print("Error Title")
                        title=""
                    try:    
                        description= my_video.description
                    except:
                        print("Error Description")
                        description=""
                    try:    
                        length= my_video.length
                    except:
                        print("Error Length")
                        length="0"

                    bugvideo=str('T')

                    try:    
                        writer.writerow([title, description, length, bugvideo])
                    except:
                        writer.writerow(["", "", "10000", "T"])

                    
                    

                csvfile.close()
            
                with open('url_video.txt', 'w') as url_video:
                    url_video.write(line)
                url_video.close()
