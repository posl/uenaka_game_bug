from dataclasses import replace
import spacy
import nltk
import nltk.sentiment.util
import json


def captionProcessing(filename, keyword):
    nlp = spacy.load('en_core_web_sm')
    
    
    with open(filename) as f:

        data = json.load(f)
        
        for element in data:
            text= element['text']
            text_processing= nlp(text.lower())

            words_text= [token.text for token in text_processing]
            lemma_text = [token.lemma_ for token in text_processing]

            text_filterd= []
            text_filterd.append(lemma_text)

            element['text']=text_filterd

            json_object = json.dumps(data)

            with open("caption_processing.json", "w") as outfile:
                outfile.write(json_object)
        f.close()  


