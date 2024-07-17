from youtube_video_info import youtubeVideo
from arff_generation import arff_file_generation
from downloadYouTube import downoaldYoutubeVideo
from caption_processing import captionProcessing
from chdir import cwd
import numpy as np
import glob
import os
import sys

jar_name = os.path.abspath("bugvideo-classifier.jar")
random_forest_model=os.path.abspath("random-forest-bugvideo.model")


def main():
    
  if len(sys.argv) < 2:
    print("Parametri mancanti. Info: [../input_link_file.txt] [output]")
    print("Sto uscendo...")
    sys.exit()
  
  else:
    input_filename = sys.argv[1]
    output= sys.argv[2]
    
    
    os.makedirs(f"results-{output}", exist_ok=True)
    with cwd(f"results-{output}"):
      youtubeVideo(input_filename)
      print("finish download INFO")                          

      for path_video in glob.glob("v*"):
        with cwd(path_video):
          arff_file_generation()
          print("finish generate file")
          input_file="isbugvideo.arff"   

          print("run jar")
          cmd = f"java -jar {jar_name} {random_forest_model} {input_file}"
          print(cmd)
          os.system(cmd)
          print("jar ok")
          out_file="output-segment.txt"
                    
          with open(out_file, "r") as out_file:
                  line = out_file.readline().strip().split()
                  # if line==[]:
                  #       out_file.close()
                  # elif line[0]==line[1]:
                  #       print(line)
                  url_video="url_video.txt"
                  downoaldYoutubeVideo(url_video)
                  print("finish download video")                          

                  # else: 
                  #       print("no bug video")

        
if __name__ == "__main__":
  main()