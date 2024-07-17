import json
import os
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from chdir import cwd
import traceback
import sys

SAVE_PATH = "video"

def downoaldYoutubeVideo(path):
    
    #create directory for each video
    with open(path, 'r') as link_file:
        lines = link_file.readlines()
        for i in range(len(lines)):
            path_video = f"download"
            if not os.path.exists(path_video):
                os.mkdir(path_video)
            
            line = lines[i] 
            try: 
                my_video = YouTube(line) 
            except: 
                print("Connection Error")

            print("********************Download video*************************")

            #Download video
            with cwd(path_video):
                try:
                    print("Getting video streams...")
                    streams = my_video.streams
                    print(f"Number of streams: {len(streams)}")
                    for stream in streams:
                        print(stream)
                except :
                    etype, value, tb = sys.exc_info()
    
                    traceback.format_exception(etype, value, tb) # list[str] を返す
                    traceback.format_tb(tb) # list[str] を返す
                    traceback.format_exception_only(etype, value) # list[str] を返す
    
                    traceback.format_exc() # tuple[str] を返す
                    print(traceback.format_exception(etype, value, tb))
            #     try:
            #         for stream in my_video.streams:
            #             print(stream)

            # #set stream resolution
            #         my_video = my_video.streams.get_highest_resolution()
            #         my_video.download(None,"video.mp4")
            #         print("path video ", path_video )
            #     except:
            #         with open("info.txt", "w") as f:
            #             f.write(line)


                print("********************Download caption*************************")

                myUrl = line.split('=')
                try:
                    transcript = YouTubeTranscriptApi.get_transcript(myUrl[1], languages=['en'])
                    with open("caption.json", "w") as f:
                        json.dump(transcript, f) 
                except: 
                    print("Error - No Caption")   

