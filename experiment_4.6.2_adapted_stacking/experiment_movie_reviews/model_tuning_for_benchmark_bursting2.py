from keras.layers import LSTM, Lambda,GlobalAveragePooling1D, Dense
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras import layers
from keras.models import Sequential, model_from_json, Model
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import Adam, RMSprop, SGD, Nadam
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import GlobalMaxPooling1D
from keras.layers.core import Activation
from keras import regularizers
from keras.regularizers import l2
import time
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import re
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
import numpy as np
from flair.data import Sentence
from flair.embeddings import WordEmbeddings
import pandas as pd
from nltk.corpus import stopwords
english_stopwords = stopwords.words('english')



with open( "additional_stopwords.txt", "r" ) as f:
    additional_stopwords = f.read().splitlines()

#english_stopwords.extend(additional_stopwords)

def get_cnn_model_to_burst_benchmark_old( sequence_len, vector_dim, num_classes ):
    model = Sequential()
    model.add( Conv1D( filters=100,input_shape=(sequence_len, vector_dim), kernel_size=3, padding='same', activation='relu', name='conv_layer_1' ))
    model.add( GlobalAveragePooling1D(data_format='channels_last') )
    model.add( Dense(128, activation='relu') )
    model.add( layers.Dropout(0.5))
    model.add( Dense( num_classes, activation='sigmoid'))
    optimizer = RMSprop()
    model.compile( loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print( model.summary() )
    return model

def get_cnn_model_to_burst_benchmark( sequence_len, vector_dim, num_classes ):
    model = Sequential()
    model.add( Conv1D( filters=500,input_shape=(sequence_len, vector_dim), kernel_size=3, padding='same', activation='relu', name='conv_layer_1' ))
    model.add( GlobalMaxPooling1D(data_format='channels_last') )
    model.add( Dense(128, activation='relu') )
    model.add( layers.Dropout(0.5))
    model.add( Dense( num_classes, activation='sigmoid'))
    #optimizer = RMSprop()
    optimizer='adam'
    model.compile( loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print( model.summary() )
    return model





def load_movie_reviews_data_from_csv( datapath ):
    df = pd.read_csv( datapath, delimiter=";")
    source_sentences = df['text'].tolist()
    label = df['label'].tolist()
    return source_sentences, label

def get_data_for_nn_meta_function( datapath, sequence_len, input_vector_dim ):
    print("Start trainings process mit dataset {}".format( datapath ) )
    source_sentences, label = load_movie_reviews_data_from_csv( datapath )
    print( "Convert sentences to vectors...")
    sentences_preprocessed = preprocess_sentences( source_sentences, sequence_len  )
    #sentences_preprocessed = source_sentences
    X_sources_sentences = get_vectors( sentences_preprocessed, sequence_len, input_vector_dim, fastText_model  )
    #y_label = np_utils.to_categorical( label )
    y_label = label
    return X_sources_sentences, y_label

def load_word_embeddings():
    print("Loading fastText embeddings...")
    #de_fastText_embeddings = WordEmbeddings('en')
    de_fastText_embeddings = WordEmbeddings('en-glove')
    return de_fastText_embeddings

def get_vectors( preprocessed_sentences, seq_len, embedding_dim,de_fastText_embeddings ):
    num_sentences = len( preprocessed_sentences )
    count_unknown_words=0

    data_matrix = np.zeros(( num_sentences,seq_len,embedding_dim ))

    for i, sentence in enumerate( preprocessed_sentences ):
        if len(sentence) > 1:
            flair_sentence = Sentence( sentence )
            de_fastText_embeddings.embed( flair_sentence)
            for j,token in enumerate(flair_sentence):
                word_embedding = token.embedding.data.numpy()
                data_matrix[i,j] = word_embedding
    return data_matrix

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

    # cut sentence length

    sentences_as_word_list = [ filter_stopwords(s) for s in sentences_as_word_list]


    sentences_as_word_list = [ s[:max_sentence_length] if len(s)> max_sentence_length else s for s in sentences_as_word_list ]



    sentence_lengths = [ len(s) for s in sentences_as_word_list]



    print("Max sentence length: ", max(sentence_lengths) )
    print("Mean sentence lenght: ", np.mean(sentence_lengths))

    sentences_strings = [ " ".join(s) for s in sentences_as_word_list ]
    return sentences_strings

if __name__ == "__main__":
    #EMBEDDING_DIM = 300
    #INPUT_VECTOR_DIM = 300
    EMBEDDING_DIM = 100
    INPUT_VECTOR_DIM = 100
    #SEQUENCE_LEN = 35
    #SEQUENCE_LEN = 100
    SEQUENCE_LEN = 50
    SEQUENCE_LEN = 200
    BATCH_SIZE = 32
    EPOCHS=10
    EPOCHS=20

    NUM_CLASSES = 1
    nn_model_1 = get_cnn_model_to_burst_benchmark( SEQUENCE_LEN,INPUT_VECTOR_DIM, NUM_CLASSES  )

    fastText_model = load_word_embeddings()

    X_train, y_train = get_data_for_nn_meta_function( "movie_reviews_train.csv", SEQUENCE_LEN, INPUT_VECTOR_DIM )

    X_test, y_test= get_data_for_nn_meta_function( "movie_reviews_test.csv", SEQUENCE_LEN, INPUT_VECTOR_DIM )

    h1 = nn_model_1.fit( X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test) )
