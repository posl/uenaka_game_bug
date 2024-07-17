from chdir import cwd
from uenaka_download_youtube import youtubeVideo
import os
import numpy as np
import sys
import glob

def main():
  
  if len(sys.argv) < 1:
    print("Parametri mancanti. Info: [input_link_file.csv] [output]")
    print("Sto uscendo...")
    sys.exit()
  
  else:
    input_filename = sys.argv[1]

  data_file=os.path.abspath(input_filename)#"input_link_file.csv"
  # os.mkdir(f"results-{output}")
  with cwd("results-uenaka_output/v0/download/segments"):
    line=youtubeVideo(data_file)
    print("finish download video")
      

if __name__ == "__main__":
  main()