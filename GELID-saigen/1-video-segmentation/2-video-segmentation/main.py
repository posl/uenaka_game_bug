from caption_processing import captionProcessing
from video_segmentation import videoTrimTrashold
from chdir import cwd
import numpy as np
import glob
import os
import sys
# import traceback


def main():
    
  if len(sys.argv) < 2:
    print("Parametri mancanti. Info: result-fold")
    print("Sto uscendo...")
    sys.exit()
  
  else:
    input_result = sys.argv[1]
    threshold = sys.argv[2]
  
  keywords=[]
  with open(os.path.abspath("generated_keyword.txt"), 'r') as keywordlist:
      lines = keywordlist.readlines()
      for i in range(len(lines)):
          line= lines[i]
          keyword=line.replace("\n","")
          keywords.append(keyword)
          
  print(keywords)
  
  for path_result in glob.glob(input_result):
    with cwd(path_result):  
      for path_video in glob.glob("v*"):
        with cwd(path_video):
          try:
            for path_segment in glob.glob("d*"):
              with cwd(path_segment):
                print("caption analysis")
                captionProcessing("caption.json")
                print("finish caption")

                print("video segmentation")
                videoTrimTrashold("caption_processing.json", keywords,threshold)
          except :
            # print(e.__class__.__name__) # ZeroDivisionError
            # print(e.args) # ('division by zero',)
            # print(e) # division by zero
            # print(f"{e.__class__.__name__}: {e}") # ZeroDivisionError: division by zero
            print("no video to analyse")
                # 発生中の例外に関する情報を取得する
            # etype, value, tb = sys.exc_info()
    
            # traceback.format_exception(etype, value, tb) # list[str] を返す
            # traceback.format_tb(tb) # list[str] を返す
            # traceback.format_exception_only(etype, value) # list[str] を返す
    
            # traceback.format_exc() # tuple[str] を返す
            # print(traceback.format_exception(etype, value, tb))
if __name__ == "__main__":
  main()