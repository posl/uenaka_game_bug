import nltk
import spacy
import gensim, warnings
import gensim.downloader as api
import pandas as pd
import numpy as geek
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from spacy.tokens import token



warnings.filterwarnings(action = 'ignore')
dataset = api.load("text8")
data = [d for d in dataset]

def tagged_document(list_of_list_of_words):
   for i, list_of_words in enumerate(list_of_list_of_words):
      yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])
data_training = list(tagged_document(data))

model = gensim.models.doc2vec.Doc2Vec(vector_size=40, min_count=2, epochs=30)

model.build_vocab(data_training)

with open('training_binary_doc2vec.arff', 'w') as feature_file:

  with open("uenaka_segments_info.csv", 'r') as f: # open in readonly mode
      subtitle_list = pd.read_csv(f)

      feature_file.write('''@RELATION classification
@ATTRIBUTE f1 REAL
@ATTRIBUTE f2 REAL
@ATTRIBUTE f3 REAL
@ATTRIBUTE f4 REAL
@ATTRIBUTE f5 REAL
@ATTRIBUTE f6 REAL
@ATTRIBUTE f7 REAL
@ATTRIBUTE f8 REAL
@ATTRIBUTE f9 REAL
@ATTRIBUTE f10 REAL
@ATTRIBUTE f11 REAL
@ATTRIBUTE f12 REAL
@ATTRIBUTE f13 REAL
@ATTRIBUTE f14 REAL
@ATTRIBUTE f15 REAL
@ATTRIBUTE f16 REAL
@ATTRIBUTE f17 REAL
@ATTRIBUTE f18 REAL
@ATTRIBUTE f19 REAL
@ATTRIBUTE f20 REAL
@ATTRIBUTE f21 REAL
@ATTRIBUTE f22 REAL
@ATTRIBUTE f23 REAL
@ATTRIBUTE f24 REAL
@ATTRIBUTE f25 REAL
@ATTRIBUTE f26 REAL
@ATTRIBUTE f27 REAL
@ATTRIBUTE f28 REAL
@ATTRIBUTE f29 REAL
@ATTRIBUTE f30 REAL
@ATTRIBUTE f31 REAL
@ATTRIBUTE f32 REAL
@ATTRIBUTE f33 REAL
@ATTRIBUTE f34 REAL
@ATTRIBUTE f35 REAL
@ATTRIBUTE f36 REAL
@ATTRIBUTE f37 REAL
@ATTRIBUTE f38 REAL
@ATTRIBUTE f39 REAL
@ATTRIBUTE f40 REAL
@ATTRIBUTE classification {presentation, logic, balance, performance, non-informative}

@DATA 
''')
    
      nlp = spacy.load('en_core_web_sm')
      stopwords = spacy.lang.en.stop_words.STOP_WORDS

      # create a Doc object
      
      for idx,row in subtitle_list.iterrows():

          subtitle=row['subtitle']
          label= row['label']
                    

          sub=subtitle.lower()
          doc = nlp(sub)
          # Generate the tokens
          tokens = [token.text for token in doc]
          print(tokens)

          # Generate lemmatized tokens
          lemmas = [token.lemma_ for token in doc]

          # Remove stopwords and non-alphabetic tokens
          a_lemmas = [lemma for lemma in lemmas if lemma.isalpha() and lemma not in stopwords]

          print(a_lemmas)

          features= model.infer_vector(a_lemmas)

          features_list = list(features)
          print(features_list)

          features_list.append(label)
          print(features_list)

          features_rf=','.join(str(e) for e in features_list)

          feature_file.write(features_rf)
          feature_file.write("\n")



