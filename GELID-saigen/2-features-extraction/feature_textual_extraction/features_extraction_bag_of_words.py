import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import spacy
import re

corpus=[]
cv = CountVectorizer(stop_words='english', lowercase=True)

with open('training_classification_BoW.arff', 'w') as feature_file:

  with open("uenaka_segments_info.csv", 'r') as f: # open in readonly mode
      subtitle_list = pd.read_csv(f)
     
      for idx,row in subtitle_list.iterrows():

          sub=row['subtitle']        
          
        
          subtitle=re.sub(r'[0-9]', ' ', sub)
          corpus.append(subtitle)    
      
      word_count = cv.fit_transform(corpus) # Fit the model

      n_features=(len(cv.get_feature_names_out())) # Print all the words in vocabulary
      
      feature_file.write('''@RELATION classification''')
      feature_file.write("\n")

      
      for n in range(n_features):
        feature_file.write('''@ATTRIBUTE f''' + str(n) + ''' REAL''' )
        feature_file.write("\n")
        
      feature_file.write('''@ATTRIBUTE classification {presentation, logic, balance, performance, non-informative}

@DATA 
''')
      for idx,row in subtitle_list.iterrows():

       features=word_count[idx].toarray()
       label=row['label']
      
       features_list = features.tolist()

       features_list.append(label)
       
       features_rf=','.join(str(e) for e in features_list).replace('[','').replace(']','')

       feature_file.write(features_rf)
       feature_file.write("\n")

