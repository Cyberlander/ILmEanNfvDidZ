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

def get_cnn_model( sequence_len, vector_dim, num_classes ):
    model = Sequential()
    model.add( Conv1D( filters=100,input_shape=(sequence_len, vector_dim), kernel_size=3, padding='same', activation='relu', name='conv_layer_1' ))
    model.add( GlobalAveragePooling1D(data_format='channels_last') )
    model.add( Dense(128, activation='relu') )
    model.add( Dense( num_classes, activation='sigmoid'))
    optimizer = RMSprop()
    model.compile( loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print( model.summary() )
    return model

def get_lstm_model( vector_dim ):
    inputs = layers.Input( shape=(None, vector_dim))
    lstm = layers.LSTM( 128 )(inputs)
    dense = layers.Dense( 1 , activation='sigmoid')(lstm)
    model = Model( inputs, dense )
    optimizer = RMSprop()
    model.compile( loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print( model.summary() )
    return model

def get_gru_model( vector_dim ):
    inputs = layers.Input( shape=(None, vector_dim))
    lstm = layers.GRU( 128 )(inputs)
    dense = layers.Dense( 1 , activation='sigmoid')(lstm)
    model = Model( inputs, dense )
    optimizer = RMSprop()
    model.compile( loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print( model.summary() )
    return model

def get_combined_model( input_shape, num_classes ):
    model = Sequential(  )
    model.add( layers.convolutional.Conv1D( 500, 2, strides=1, input_shape=input_shape ) )
    model.add( layers.convolutional.MaxPooling1D( pool_size=2))
    model.add( layers.Bidirectional(layers.LSTM(100, dropout=0.2)))
    model.add( layers.Dense(num_classes, activation='softmax',
    kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))
    optimizer = RMSprop()
    model.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print( model.summary() )
    return model

def save_model_architecture( nn_model, path ):
    model_json = nn_model.to_json()
    with open( path , "w") as json_file:
        json_file.write( model_json )
