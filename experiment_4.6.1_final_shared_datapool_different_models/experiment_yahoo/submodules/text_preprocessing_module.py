import re
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords
english_stopwords = stopwords.words('english')

def filter_stopwords( sentence ):
    sentence_filtered = []
    for word in sentence:
        if word not in english_stopwords:
            sentence_filtered.append( word )
    return sentence_filtered


def preprocess_sentences( sentences, max_sentence_length ):
    found = 0
    found_umlaute = 0
    found_tags = 0
    for i,s in enumerate( sentences ):
        if re.search('DE\d{2}[ ]\d{4}[ ]\d{4}[ ]\d{4}[ ]\d{4}[ ]\d{2}|DE\d{20}', s):
            found += 1
            sentences[i] = re.sub( 'DE\d{2}[ ]\d{4}[ ]\d{4}[ ]\d{4}[ ]\d{4}[ ]\d{2}|DE\d{20}', "Kontonummer", sentences[i] )
        if re.search('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', s):
            found += 1
            sentences[i] = re.sub( 'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', "URL", sentences[i] )

        if re.search(' % ', s):
            found += 1
            sentences[i] = re.sub( ' % ' , " Prozent ", sentences[i] )

        if re.search(' § ', s):
            found += 1
            sentences[i] = re.sub( ' § ' , " Paragraph ", sentences[i] )
        if re.search(' §§ ', s):
            found += 1
            sentences[i] = re.sub( ' §§ ' , " Paragraph ", sentences[i] )
        if re.search('i\.S\.d\.', s):
            found += 1
            sentences[i] = re.sub( 'i\.S\.d\.' , "im Sinne der", sentences[i] )
        if re.search('[A-Za-z]*straße', s):
            found += 1
            sentences[i] = re.sub( '[A-Za-z]*straße' , "Strasse", sentences[i] )


    sentences_as_word_list = [ text_to_word_sequence(str(s), lower=False) for s in sentences]

    #sentences_as_word_list = [ filter_stopwords(s) for s in sentences_as_word_list]

    # cut sentence length
    sentences_as_word_list = [ s[:max_sentence_length] if len(s)> max_sentence_length else s for s in sentences_as_word_list ]

    sentence_lengths = [ len(s) for s in sentences_as_word_list]

    print("Max sentence length: ", max(sentence_lengths) )
    print("Mean sentence lenght: ", np.mean(sentence_lengths))

    sentences_strings = [ " ".join(s) for s in sentences_as_word_list ]
    return sentences_strings
