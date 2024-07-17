from dataclasses import dataclass
from chdir import cwd
from download_youtube import youtubeVideo
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

  data_file=os.path.abspath(input_filename)
  # os.makedirs(f"results-{output}", exist_ok=True)
  with cwd("results-uenaka_output/v0/download/segments"):
    youtubeVideo(data_file)
    print("finish download video")
      

if __name__ == "__main__":
  main()