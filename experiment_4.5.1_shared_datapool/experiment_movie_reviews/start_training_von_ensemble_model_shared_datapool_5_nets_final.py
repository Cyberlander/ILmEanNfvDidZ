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
    #CHUNK_DATABASE_PATH = "../../datasets_for_experiments/movie_reviews_20_chunks/movie_reviews_chunk_{}.csv"
    CHUNK_DATABASE_PATH = "../../datasets_for_experiments/movie_reviews_20_growing_dataset/movie_reviews_growing_dataset_{}.csv"
    # yahoo_data_sentiment_classification_chunk_0.csv
    EMBEDDING_DIM = 300
    INPUT_VECTOR_DIM = 300
    SEQUENCE_LEN = 35  #(=max_sentence_length)
    BATCH_SIZE = 64

    NUM_CLASSES = 1
    INPUT_SHAPE = ( SEQUENCE_LEN, EMBEDDING_DIM )
    EPOCHS = experiment_config.EPOCHS


    nn_model_1 = neural_network_module.get_cnn_model( SEQUENCE_LEN,INPUT_VECTOR_DIM, NUM_CLASSES  )
    nn_model_2 = neural_network_module.get_cnn_model( SEQUENCE_LEN,INPUT_VECTOR_DIM, NUM_CLASSES  )
    nn_model_3 = neural_network_module.get_cnn_model( SEQUENCE_LEN,INPUT_VECTOR_DIM, NUM_CLASSES  )
    nn_model_4 = neural_network_module.get_cnn_model( SEQUENCE_LEN,INPUT_VECTOR_DIM, NUM_CLASSES  )
    nn_model_5 = neural_network_module.get_cnn_model( SEQUENCE_LEN,INPUT_VECTOR_DIM, NUM_CLASSES  )


    meta_model = ensemble_module.get_meta_classifier()

    fastText_model = word_embedding_module.load_word_embeddings()


    X_to_be_predicted = None
    y_to_be_predicted = None

    # Erste Lösung, wir sammeln alle Holdouts und trainieren am Ende damit den Meta-Classifier
    training_time_overall = 0
    #for i in range(1):

    ACCURACIES = []
    PRECISIONS = []
    RECALLS = []
    F1S = []
    X_sources_sentences_complete, y_label_complete = get_data_for_nn_meta_function( "../../datasets_for_experiments/movie_reviews_20_growing_dataset/movie_reviews_complete_datapool_shuffled.csv", SEQUENCE_LEN, INPUT_VECTOR_DIM )
    X_TEST, y_TEST = get_data_for_nn_meta_function( "../../datasets_for_experiments/movie_reviews_20_chunks/movie_reviews_chunk_test.csv", SEQUENCE_LEN, INPUT_VECTOR_DIM )
    for i in range(20):
        print("\n--------------------[ {} ]--------------------".format( i ))


        current_threshold = (i+1)*2000
        if i==19:
            X_sources_sentences = X_sources_sentences_complete
            y_label = y_label_complete
        else:
            X_sources_sentences = X_sources_sentences_complete[:current_threshold]
            y_label = y_label_complete[:current_threshold]

        y_label = np.array( y_label )

        y_label = np.array( y_label )



        _, X_holdout, _, y_holdout = train_test_split( X_sources_sentences, y_label, test_size=0.2, random_state=42)


        X_to_be_predicted = X_holdout
        #y_to_be_predicted = categorical_to_numeric( y_holdout_1 )
        y_to_be_predicted = y_holdout


        start_time = time.time()


        kf = KFold(n_splits=5)
        for i, (train_index, test_index) in enumerate(kf.split(X_sources_sentences)):
            X_train, y_train = X_sources_sentences[train_index], y_label[train_index]
            if i == 0:
                h1 = nn_model_1.fit( X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE )
            elif i == 1:
                h2 = nn_model_2.fit( X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE )
            elif i == 2:
                h3 = nn_model_3.fit( X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE )
            elif i == 3:
                h4 = nn_model_4.fit( X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE )
            elif i == 4:
                h5 = nn_model_5.fit( X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE )



        end_time = time.time()
        training_time_overall += (end_time-start_time)
        print("\nTraining {} epochs took {} seconds.".format( EPOCHS, end_time-start_time))

        print("Instant meta training and testing...")
        X_train_meta_1 = nn_model_1.predict( X_to_be_predicted )
        X_train_meta_2 = nn_model_2.predict( X_to_be_predicted )
        X_train_meta_3 = nn_model_3.predict( X_to_be_predicted )
        X_train_meta_4 = nn_model_4.predict( X_to_be_predicted )
        X_train_meta_5 = nn_model_5.predict( X_to_be_predicted )
        X_train_meta = np.concatenate( ( X_train_meta_1, X_train_meta_2, X_train_meta_3, X_train_meta_4, X_train_meta_5 ), axis=1 )
        y_train_meta = y_to_be_predicted
        meta_model = meta_model.fit(X_train_meta,y_train_meta)

        X_test_meta_1 = nn_model_1.predict( X_TEST )
        X_test_meta_2 = nn_model_2.predict( X_TEST )
        X_test_meta_3 = nn_model_3.predict( X_TEST )
        X_test_meta_4 = nn_model_4.predict( X_TEST )
        X_test_meta_5 = nn_model_5.predict( X_TEST )
        X_test_meta = np.concatenate( ( X_test_meta_1, X_test_meta_2, X_test_meta_3, X_test_meta_4, X_test_meta_5 ), axis=1 )
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

    with open( "verlauf_5_nets_testset_1_final.txt", "w" ) as f:
        f.write(str(ACCURACIES))
        f.write(str(PRECISIONS))
        f.write(str(RECALLS))
        f.write(str(F1S))

    #with open("current_output_ensemble.txt", "w" ) as f:
        #f.write()
