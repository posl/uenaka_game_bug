from dataclasses import replace
import spacy
import nltk
import nltk.sentiment.util
import json


def captionProcessing(filename):

# Opening JSON file
    nlp = spacy.load('en_core_web_sm')
    
    with open(filename) as f:
        # returns JSON object as 
        # a dictionary
        data = json.load(f)
        
        # Iterating through the json
        # list
        for element in data:
            text= element['text']
            text_processing= nlp(text.lower())

            words_text= [token.text for token in text_processing]
            lemma_text = [token.lemma_ for token in text_processing]

            #tokens_neg_marked = nltk.sentiment.util.mark_negation(lemma_text)  

            #caption_processing = [t for t in tokens_neg_marked
                        #if t.replace("_NEG", "").isalnum() and
                        #t.replace("_NEG", "")]

            text_filterd= []
            text_filterd.append(lemma_text)

            element['text']=text_filterd

            json_object = json.dumps(data)

            with open("caption_processing.json", "w") as outfile:
                outfile.write(json_object)
        # Closing file
        f.close()  


