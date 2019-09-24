import time
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from submodules import (data_io, neural_network_module,
    word_embedding_module, text_preprocessing_module, ensemble_module,  experiment_config, metrics )
from sklearn.model_selection import KFold
from flair.embeddings import WordEmbeddings
from nltk.corpus import stopwords
import re
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
english_stopwords = stopwords.words('english')
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
import pandas as pd
def get_lstm_model( vector_dim ):
    inputs = layers.Input( shape=(None, vector_dim))
    lstm = layers.LSTM( 64 )(inputs)
    dense = layers.Dense( 1 , activation='sigmoid')(lstm)
    model = Model( inputs, dense )
    optimizer = RMSprop()
    model.compile( loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print( model.summary() )
    return model

def get_gru_model( vector_dim ):
    inputs = layers.Input( shape=(None, vector_dim))
    lstm = layers.GRU( 64 )(inputs)
    dense = layers.Dense( 1 , activation='sigmoid')(lstm)
    model = Model( inputs, dense )
    optimizer = RMSprop()
    model.compile( loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print( model.summary() )
    return model

def get_data_for_nn_meta_function( datapath, sequence_len, input_vector_dim ):
    print("Start trainings process mit dataset {}".format( datapath ) )
    source_sentences, label = data_io.load_movie_reviews_data_from_csv( datapath )
    print( "Convert sentences to vectors...")
    sentences_preprocessed = preprocess_sentences( source_sentences, sequence_len  )
    X_sources_sentences = word_embedding_module.get_vectors( sentences_preprocessed, sequence_len, input_vector_dim, fastText_model  )
    #y_label = np_utils.to_categorical( label )
    y_label = label
    return X_sources_sentences, y_label

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

def load_word_embeddings():
    print("Loading fastText embeddings...")
    #de_fastText_embeddings = WordEmbeddings('en')
    de_fastText_embeddings = WordEmbeddings('en-glove')
    return de_fastText_embeddings

def categorical_to_numeric( y_data ):
    y_numeric = []
    for element in y_data:
        y_numeric.append( np.argmax( element) )
    return y_numeric

if __name__ == "__main__":
    #DATABASE_PATH = 'db/yahoo_data_sentiment_classification.csv'
    #CHUNK_DATABASE_PATH = "../../datasets_for_experiments/movie_reviews_20_chunks/movie_reviews_chunk_{}.csv"
    CHUNK_DATABASE_PATH = "../../datasets_for_experiments/movie_reviews_10_to_burst_benchmark/movie_reviews_datapool_to_burst_benchmark{}.csv"
    # yahoo_data_sentiment_classification_chunk_0.csv
    EMBEDDING_DIM = 100
    INPUT_VECTOR_DIM = 100
    #EMBEDDING_DIM = 300
    #INPUT_VECTOR_DIM = 300
    SEQUENCE_LEN = 100  #(=max_sentence_length)
    SEQUENCE_LEN = 200
    #SEQUENCE_LEN = 120
    BATCH_SIZE = 32

    NUM_CLASSES = 1
    INPUT_SHAPE = ( SEQUENCE_LEN, EMBEDDING_DIM )
    EPOCHS = experiment_config.EPOCHS
    EPOCHS = 10
    EPOCHS = 5


    nn_model_1 = neural_network_module.get_cnn_model_to_burst_benchmark( SEQUENCE_LEN,INPUT_VECTOR_DIM, NUM_CLASSES  )
    #nn_model_2 = neural_network_module.get_cnn_model_to_burst_benchmark( SEQUENCE_LEN,INPUT_VECTOR_DIM, NUM_CLASSES  )
    nn_model_2 = get_lstm_model( INPUT_VECTOR_DIM )
    #nn_model_3 = neural_network_module.get_cnn_model_to_burst_benchmark( SEQUENCE_LEN,INPUT_VECTOR_DIM, NUM_CLASSES  )
    nn_model_3 = get_gru_model( INPUT_VECTOR_DIM )

    meta_model = ensemble_module.get_meta_classifier_to_bust_benchmark()

    fastText_model = load_word_embeddings()


    X_to_be_predicted = None
    y_to_be_predicted = None

    # Erste Lösung, wir sammeln alle Holdouts und trainieren am Ende damit den Meta-Classifier
    training_time_overall = 0
    #for i in range(1):

    ACCURACIES = []
    PRECISIONS = []
    RECALLS = []
    F1S = []
    X_sources_sentences_complete, y_label_complete = get_data_for_nn_meta_function(  "../../datasets_for_experiments/movie_reviews_10_to_burst_benchmark/movie_reviews_train_shuffled_short_last_chunk.csv", SEQUENCE_LEN, INPUT_VECTOR_DIM )
    X_TEST, y_TEST = get_data_for_nn_meta_function( "../../datasets_for_experiments/movie_reviews_10_to_burst_benchmark/movie_reviews_test.csv", SEQUENCE_LEN, INPUT_VECTOR_DIM )
    for i in range(10):
        print("\n--------------------[ {} ]--------------------".format( i ))


        current_threshold = (i+1)*2500
        if i==9:
            X_sources_sentences = X_sources_sentences_complete
            y_label = y_label_complete
        else:
            X_sources_sentences = X_sources_sentences_complete[:current_threshold]
            y_label = y_label_complete[:current_threshold]

        y_label = np.array( y_label )



        #_, X_holdout, _, y_holdout = train_test_split( X_sources_sentences, y_label, test_size=0.2, random_state=42)
        _, X_holdout, _, y_holdout = train_test_split( X_sources_sentences, y_label, test_size=0.90, random_state=42)


        #X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split( X_sources_sentences, y_label, test_size=0.5, random_state=42)


        # das sammeln der Hold-outs von NN 1 reicht, denn es sieht alle Datenchunks
        if i == 0:
            X_to_be_predicted = X_holdout
            #y_to_be_predicted = categorical_to_numeric( y_holdout_1 )
            y_to_be_predicted = y_holdout
        else:
            X_to_be_predicted = np.concatenate( ( X_to_be_predicted, X_holdout ), axis=0 )
            #y_tmp = categorical_to_numeric(  y_holdout_1 )
            y_tmp = y_holdout
            y_to_be_predicted = np.concatenate( ( y_to_be_predicted, y_tmp), axis=0 )




        kf = KFold(n_splits=3)
        for i, (train_index, test_index) in enumerate(kf.split(X_sources_sentences)):
            X_train, y_train = X_sources_sentences[train_index], y_label[train_index]
            if i == 0:
                h1 = nn_model_1.fit( X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE )
            elif i == 1:
                h2 = nn_model_2.fit( X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE )
            elif i == 2:
                h3 = nn_model_3.fit( X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE )



        print("Instant meta training and testing...")
        X_train_meta_1 = nn_model_1.predict( X_to_be_predicted )
        X_train_meta_2 = nn_model_2.predict( X_to_be_predicted )
        X_train_meta_3 = nn_model_3.predict( X_to_be_predicted )
        X_train_meta = np.concatenate( ( X_train_meta_1, X_train_meta_2, X_train_meta_3 ), axis=1 )
        y_train_meta = y_to_be_predicted
        meta_model = meta_model.fit(X_train_meta,y_train_meta)

        X_test_meta_1 = nn_model_1.predict( X_TEST )
        X_test_meta_2 = nn_model_2.predict( X_TEST )
        X_test_meta_3 = nn_model_3.predict( X_TEST )
        X_test_meta = np.concatenate( ( X_test_meta_1, X_test_meta_2, X_test_meta_3  ), axis=1 )
        #y_test_meta = categorical_to_numeric( y_label )
        y_test_meta = y_TEST

        meta_predictions = meta_model.predict( X_test_meta )
        #acc = accuracy_score( y_test_meta, meta_predictions )

        acc, pre, rec, f1 = metrics.compute_metrics( y_test_meta, meta_predictions )
        print("Meta accuracy: ", acc )
        print("Meta precision: ", pre )
        print("Meta recall: ", rec )
        print("Meta f1: ", f1 )

        ACCURACIES.append( acc )
        PRECISIONS.append( pre )
        RECALLS.append( rec )
        F1S.append( f1)

    print("\nAccuracies")
    print(ACCURACIES)
    print(np.mean(ACCURACIES))
    print("\nPrecisions")
    print(PRECISIONS)
    print(np.mean(PRECISIONS))
    print("\nRecalls")
    print(RECALLS)
    print(np.mean(RECALLS))
    print("\nF1s")
    print(F1S)
    print(np.mean(F1S))

    with open( "verlauf_3_nets_testset_1_to_burst_benchmark_last_chunk.txt", "w" ) as f:
        f.write(str(ACCURACIES))
        f.write(str(PRECISIONS))
        f.write(str(RECALLS))
        f.write(str(F1S))

    #with open("current_output_ensemble.txt", "w" ) as f:
        #f.write()
