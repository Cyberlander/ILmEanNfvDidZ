import time
from numpy.random import seed
import pandas as pd
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from submodules import (data_io, neural_network_module,
    word_embedding_module, text_preprocessing_module, ensemble_module,  experiment_config,metrics)
from sklearn.model_selection import KFold

def get_data_for_nn_meta_function( datapath, sequence_len, input_vector_dim ):
    print("Start trainings process mit dataset {}".format( datapath ) )
    source_sentences, label = data_io.load_facebook_data_from_csv( datapath )
    print( "Convert sentences to vectors...")
    sentences_preprocessed = text_preprocessing_module.preprocess_sentences( source_sentences, sequence_len  )
    X_sources_sentences = word_embedding_module.get_vectors( sentences_preprocessed, sequence_len, input_vector_dim, fastText_model  )
    y_label = np_utils.to_categorical( label )
    return X_sources_sentences, y_label

def categorical_to_numeric( y_data ):
    y_numeric = []
    for element in y_data:
        y_numeric.append( np.argmax( element) )
    return y_numeric

def fill_evaluation_matrix( evaluation_matrix,  acc, precisions, recalls, f1s):
    print(precisions)
    evaluation_matrix[i-1][0] = acc
    evaluation_matrix[i-1][1] = precisions[0]
    evaluation_matrix[i-1][2] = recalls[0]
    evaluation_matrix[i-1][3] = f1s[0]
    evaluation_matrix[i-1][4] = precisions[1]
    evaluation_matrix[i-1][5] = recalls[1]
    evaluation_matrix[i-1][6] = f1s[1]
    evaluation_matrix[i-1][7] = precisions[2]
    evaluation_matrix[i-1][8] = recalls[2]
    evaluation_matrix[i-1][9] = f1s[2]
    return evaluation_matrix

if __name__ == "__main__":
    #DATABASE_PATH = 'db/yahoo_data_sentiment_classification.csv'
    DATAPOOL_DATABASE_PATH = "../../datasets_for_experiments/facebook_chunks/fb_datapool_{}.csv"
    # yahoo_data_sentiment_classification_chunk_0.csv
    EMBEDDING_DIM = 300
    INPUT_VECTOR_DIM = 300
    SEQUENCE_LEN = 35  #(=max_sentence_length)
    BATCH_SIZE = 64

    NUM_CLASSES = 3
    INPUT_SHAPE = ( SEQUENCE_LEN, EMBEDDING_DIM )
    EPOCHS = experiment_config.EPOCHS


    nn_model_1 = neural_network_module.get_cnn_model( SEQUENCE_LEN,INPUT_VECTOR_DIM, NUM_CLASSES  )
    nn_model_2 = neural_network_module.get_lstm_model( INPUT_VECTOR_DIM )
    nn_model_3 = neural_network_module.get_gru_model( INPUT_VECTOR_DIM  )
    nn_model_4 = neural_network_module.get_cnn_model( SEQUENCE_LEN,INPUT_VECTOR_DIM, NUM_CLASSES  )
    nn_model_5 = neural_network_module.get_lstm_model( INPUT_VECTOR_DIM )
    meta_model = ensemble_module.get_meta_classifier()

    fastText_model = word_embedding_module.load_word_embeddings()


    X_to_be_predicted = None
    y_to_be_predicted = None

    # Erste Lösung, wir sammeln alle Holdouts und trainieren am Ende damit den Meta-Classifier
    training_time_overall = 0
    ACCURACIES = []
    evaluation_matrix = np.zeros((50,10))
    for i in range(1,51):
        print("\n--------------------[ {} ]--------------------".format( i ))

        X_sources_sentences, y_label = get_data_for_nn_meta_function( DATAPOOL_DATABASE_PATH.format(i), SEQUENCE_LEN, INPUT_VECTOR_DIM )
        #X_TEST, y_TEST = get_data_for_nn_meta_function( "../../datasets_for_experiments/facebook_chunks/fb_chunk_{}.csv".format(i+1), SEQUENCE_LEN, INPUT_VECTOR_DIM)
        X_TEST, y_TEST = get_data_for_nn_meta_function( "../../datasets_for_experiments/facebook_chunks/fb_chunk_{}.csv".format(i+1), SEQUENCE_LEN, INPUT_VECTOR_DIM)



        HOLDOUT_PERCENT = 0.2
        X_holdout = X_sources_sentences
        y_holdout =  y_label



        X_to_be_predicted = X_holdout
        y_to_be_predicted = categorical_to_numeric(  y_holdout )




        start_time = time.time()
        kf = KFold(n_splits=5)
        for j, (train_index, test_index) in enumerate(kf.split(X_sources_sentences)):
            X_train, y_train = X_sources_sentences[train_index], y_label[train_index]
            if j == 0:
                h1 = nn_model_1.fit( X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE )
            elif j == 1:
                h2 = nn_model_2.fit( X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE )
            elif j == 2:
                h3 = nn_model_3.fit( X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE )
            elif j == 3:
                h3 = nn_model_4.fit( X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE )
            elif j == 4:
                h3 = nn_model_5.fit( X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE )


        end_time = time.time()
        training_time_overall += (end_time-start_time)
        print("\nTraining {} epochs took {} seconds.".format( EPOCHS, end_time-start_time))


        print("Instant meta training and testing...")
        #if i < 5:
            #X_TEST, y_TEST = get_data_for_nn_meta_function( '../../datasets_for_experiments/facebook_data_6_chunks/facebook_data_chunk_{}.csv', SEQUENCE_LEN, INPUT_VECTOR_DIM )
        #else:
        #    X_TEST, y_TEST = get_data_for_nn_meta_function( '../../datasets_for_experiments/facebook_data_6_chunks/facebook_data_test.csv', SEQUENCE_LEN, INPUT_VECTOR_DIM )

        X_train_meta_1 = nn_model_1.predict( X_to_be_predicted )
        X_train_meta_2 = nn_model_2.predict( X_to_be_predicted )
        X_train_meta_3 = nn_model_3.predict( X_to_be_predicted )
        X_train_meta_4 = nn_model_4.predict( X_to_be_predicted )
        X_train_meta_5 = nn_model_5.predict( X_to_be_predicted )
        X_train_meta = np.concatenate( ( X_train_meta_1, X_train_meta_2, X_train_meta_3, X_train_meta_4, X_train_meta_5  ), axis=1 )
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
        y_test_meta = categorical_to_numeric( y_test_meta )
        acc = accuracy_score( y_test_meta, meta_predictions )
        print("Meta accuracy: ", acc )

        precisions, recalls, f1s = metrics.compute_metrics_for_each_class(y_test_meta, meta_predictions )


        evaluation_matrix = fill_evaluation_matrix( evaluation_matrix, acc, precisions, recalls, f1s )

        print("Evaluation_matrix:")
        print(evaluation_matrix[i-1])

        ACCURACIES.append( acc )

    print("\nAccuracies")
    print(ACCURACIES)
    print(np.mean(ACCURACIES))

    #with open( "verlauf_yahoo_5_nets_next_timestep_1.txt", "w" ) as f:
        #f.write(str(ACCURACIES))

    df_evaluation_matrix = pd.DataFrame(evaluation_matrix, columns=['accuracy', 'precision_class_0', 'recall_class_0', 'f1_class_0','precision_class_1', 'recall_class_1', 'f1_class_1',
      'precision_class_2', 'recall_class_2', 'f1_class_2' ])
    df_evaluation_matrix.to_csv( "verlauf_facebook_5_nets_next_timestep_1.csv", sep=";", index=None )
