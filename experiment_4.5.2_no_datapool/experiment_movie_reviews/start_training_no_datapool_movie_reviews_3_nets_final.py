import time
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from submodules import (data_io, neural_network_module, metrics,
    word_embedding_module, text_preprocessing_module, ensemble_module,  experiment_config)

def get_data_for_nn_meta_function( datapath, sequence_len, input_vector_dim ):
    print("Start trainings process mit dataset {}".format( datapath ) )
    source_sentences, label = data_io.load_movie_reviews_data_from_csv( datapath )
    print( "Convert sentences to vectors...")
    sentences_preprocessed = text_preprocessing_module.preprocess_sentences( source_sentences, sequence_len  )
    X_sources_sentences = word_embedding_module.get_vectors( sentences_preprocessed, sequence_len, input_vector_dim, fastText_model  )
    #y_label = np_utils.to_categorical( label )
    y_label = label
    return X_sources_sentences, y_label

def categorical_to_numeric( y_data ):
    y_numeric = []
    for element in y_data:
        y_numeric.append( np.argmax( element) )
    return y_numeric

if __name__ == "__main__":
    #DATABASE_PATH = 'db/yahoo_data_sentiment_classification.csv'
    CHUNK_DATABASE_PATH = "../../datasets_for_experiments/movie_reviews_20_chunks/movie_reviews_chunk_{}.csv"
    # yahoo_data_sentiment_classification_chunk_0.csv
    EMBEDDING_DIM = 300
    INPUT_VECTOR_DIM = 300
    SEQUENCE_LEN = 35  #(=max_sentence_length)
    BATCH_SIZE = 64

    NUM_CLASSES = 1
    INPUT_SHAPE = ( SEQUENCE_LEN, EMBEDDING_DIM )
    EPOCHS = experiment_config.EPOCHS
    EPOCHS = 3


    nn_model_1 = neural_network_module.get_cnn_model( SEQUENCE_LEN,INPUT_VECTOR_DIM, NUM_CLASSES  )
    nn_model_2 = neural_network_module.get_cnn_model( SEQUENCE_LEN,INPUT_VECTOR_DIM, NUM_CLASSES  )
    nn_model_3 = neural_network_module.get_cnn_model( SEQUENCE_LEN,INPUT_VECTOR_DIM, NUM_CLASSES  )
    meta_model = ensemble_module.get_meta_classifier()

    fastText_model = word_embedding_module.load_word_embeddings()


    X_to_be_predicted = None
    y_to_be_predicted = None

    # Erste Lösung, wir sammeln alle Holdouts und trainieren am Ende damit den Meta-Classifier
    training_time_overall = 0
    ACCURACIES = []
    PRECISIONS = []
    RECALLS = []
    F1S = []
    X_TEST, y_TEST = get_data_for_nn_meta_function( "../../datasets_for_experiments/movie_reviews_20_chunks/movie_reviews_chunk_test.csv", SEQUENCE_LEN, INPUT_VECTOR_DIM )
    for i in range(20):
        print("\n--------------------[ {} ]--------------------".format( i ))
        if i == 0:
            X_sources_sentences_1, y_label_1 = get_data_for_nn_meta_function( CHUNK_DATABASE_PATH.format(i), SEQUENCE_LEN, INPUT_VECTOR_DIM )
            X_sources_sentences_2, y_label_2 = get_data_for_nn_meta_function( CHUNK_DATABASE_PATH.format(i), SEQUENCE_LEN, INPUT_VECTOR_DIM )
            X_sources_sentences_3, y_label_3 = get_data_for_nn_meta_function( CHUNK_DATABASE_PATH.format(i), SEQUENCE_LEN, INPUT_VECTOR_DIM )
        elif i == 1:
            X_sources_sentences_1, y_label_1 = get_data_for_nn_meta_function( CHUNK_DATABASE_PATH.format(i), SEQUENCE_LEN, INPUT_VECTOR_DIM )
            X_sources_sentences_2, y_label_2 = get_data_for_nn_meta_function( CHUNK_DATABASE_PATH.format(i-1), SEQUENCE_LEN, INPUT_VECTOR_DIM )
            X_sources_sentences_3, y_label_3 = get_data_for_nn_meta_function( CHUNK_DATABASE_PATH.format(i-1), SEQUENCE_LEN, INPUT_VECTOR_DIM )
        else:
            X_sources_sentences_1, y_label_1 = get_data_for_nn_meta_function( CHUNK_DATABASE_PATH.format(i), SEQUENCE_LEN, INPUT_VECTOR_DIM )
            X_sources_sentences_2, y_label_2 = get_data_for_nn_meta_function( CHUNK_DATABASE_PATH.format(i-1), SEQUENCE_LEN, INPUT_VECTOR_DIM )
            X_sources_sentences_3, y_label_3 = get_data_for_nn_meta_function( CHUNK_DATABASE_PATH.format(i-2), SEQUENCE_LEN, INPUT_VECTOR_DIM )



        X_train_1, X_holdout_1, y_train_1, y_holdout_1 = train_test_split( X_sources_sentences_1, y_label_1, test_size=0.2, random_state=42)


        # das sammeln der Hold-outs von NN 1 reicht, denn es sieht alle Datenchunks
        if i == 0:
            X_to_be_predicted = X_holdout_1
            #y_to_be_predicted = categorical_to_numeric( y_holdout_1 )
            y_to_be_predicted = y_holdout_1
        else:
            X_to_be_predicted = np.concatenate( ( X_to_be_predicted, X_holdout_1 ), axis=0 )
            #y_tmp = categorical_to_numeric(  y_holdout_1 )
            y_tmp = y_holdout_1
            y_to_be_predicted = np.concatenate( ( y_to_be_predicted, y_tmp), axis=0 )

        start_time = time.time()
        h1 = nn_model_1.fit( X_sources_sentences_1, y_label_1, epochs=EPOCHS, batch_size=BATCH_SIZE )
        h2 = nn_model_2.fit( X_sources_sentences_2, y_label_2, epochs=EPOCHS, batch_size=BATCH_SIZE )
        h3 = nn_model_3.fit( X_sources_sentences_3, y_label_3, epochs=EPOCHS, batch_size=BATCH_SIZE )
        end_time = time.time()
        training_time_overall += (end_time-start_time)
        print("\nTraining {} epochs took {} seconds.".format( EPOCHS, end_time-start_time))

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
        X_test_meta = np.concatenate( ( X_test_meta_1, X_test_meta_2, X_test_meta_3 ), axis=1 )
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

    with open( "no_datapool_verlauf_3_nets_1_testset.txt", "w" ) as f:
        f.write(str(ACCURACIES))
        f.write(str(PRECISIONS))
        f.write(str(RECALLS))
        f.write(str(F1S))
