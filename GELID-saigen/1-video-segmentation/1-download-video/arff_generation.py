import csv
import cv2
import numpy as np
import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer


def arff_file_generation():

    nlp = spacy.load('en_core_web_sm')
    stopwords = spacy.lang.en.stop_words.STOP_WORDS

    with open('isbugvideo.arff', 'w') as bugvideofile:

        with open("youtube_video_info.csv", 'r') as f:
            bugvideo_analysis = pd.read_csv(f)
            bugvideofile.write('''@RELATION bugvideo
@ATTRIBUTE n_keyword_title INTEGER
@ATTRIBUTE n_keyword_description INTEGER
@ATTRIBUTE lenght_description INTEGER
@ATTRIBUTE lenght_title INTEGER
@ATTRIBUTE lenght_video INTEGER
@ATTRIBUTE isbugvideo {true, false}

@DATA 
''')
            keywordlist=['bug','glitch','hack','hacker','cheat','heater']
            for idx,row in bugvideo_analysis.iterrows():
                #TITLE
                n_keyword_title=0
                title=str(row['youtube_title'])
                # create a Doc object
                t = nlp(title.lower())
                words_title= [token.text for token in t]
                lemma_title = [token.lemma_ for token in t]
                title_filterd= []
                for w in lemma_title:
                    if w not in stopwords:
                        title_filterd.append(w)
                    if w in keywordlist:
                        n_keyword_title = n_keyword_title +1
                keyword_title=str(n_keyword_title)
                length_title=str(len(title_filterd))
                
                #DESCRIPTION
                n_keyword_description=0
                description=str(row['youtube_description'])
                # create a Doc object
                d = nlp(description.lower())
                try:
                    words_description= [token.text for token in d]
                    lemma_description = [token.lemma_ for token in d]
                except:
                    words_description=""
                    lemma_description=""
                description_filterd= []
                for w_description in lemma_description:
                    if w_description not in stopwords:
                        description_filterd.append(w_description)
                    if w_description in keywordlist:
                        n_keyword_description = n_keyword_description +1
                keyword_description=str(n_keyword_description)
                length_description=str(len(description_filterd))

                
                #LENGTH VIDEO
                length_video=str(row['length'])
                
                bugvideo=row['is_bug_video']
                isbugvideo=bugvideo.replace('T','true')

                line=keyword_title, keyword_description, length_description, length_title, length_video, isbugvideo
                
                sent_new = line
                sentence=",".join(sent_new)
                bugvideofile.write(sentence)
                bugvideofile.write("\n")
    bugvideofile.close()
