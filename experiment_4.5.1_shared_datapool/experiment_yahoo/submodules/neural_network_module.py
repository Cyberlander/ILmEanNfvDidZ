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
    model.add( Dense( num_classes, activation='softmax'))
    optimizer = RMSprop()
    model.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print( model.summary() )
    return model

def save_model_architecture( nn_model, path ):
    model_json = nn_model.to_json()
    with open( path , "w") as json_file:
        json_file.write( model_json )
