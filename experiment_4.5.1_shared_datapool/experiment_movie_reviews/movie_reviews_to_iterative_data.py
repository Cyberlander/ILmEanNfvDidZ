import os
import pandas as pd

def texts_to_chunk_lists( texts_positive, texts_negative, MOVIE_REVIEWS_CHUNKS_PATH ):
    chunk_lists = []
    for i in range(20):
        start = i * 1000
        end = start + 1000
        list_chunk_positive = texts_positive[start:end]
        label_positive = [1]*len(list_chunk_positive)

        list_chunk_negative = texts_negative[start:end]
        label_negative = [0]*len(list_chunk_negative)

        chunk_texts = list_chunk_positive + list_chunk_negative
        chunk_labels = label_positive + label_negative
        df = pd.DataFrame( { "text":chunk_texts, "label":chunk_labels } )
        dst_path = MOVIE_REVIEWS_CHUNKS_PATH + "/movie_reviews_chunk_{}.csv".format(i)
        df.to_csv( dst_path, sep=";", index=None )

    test_chunk_positive = texts_positive[20000:]
    label_positive = [1]*len(test_chunk_positive)

    test_chunk_negative = texts_negative[20000:]
    label_negative = [0]*len(test_chunk_negative)

    chunk_texts = test_chunk_positive + test_chunk_negative
    chunk_labels = label_positive + label_negative

    df = pd.DataFrame( { "text":chunk_texts, "label":chunk_labels } )
    dst_path = MOVIE_REVIEWS_CHUNKS_PATH + "/movie_reviews_chunk_test.csv".format(i)
    df.to_csv( dst_path, sep=";", index=None )



if __name__ == "__main__":
    MOVIE_REVIEWS_CHUNKS_PATH = "movie_reviews_chunks"
    df = pd.read_csv( "movie_reviews.csv", delimiter=";", encoding="utf-8" )
    df_positive = df[ df['label'] == 1]
    df_negative = df[ df['label'] == 0]

    texts_positive = df_positive['text'].tolist()
    texts_negative = df_negative['text'].tolist()

    try:
        os.mkdir( MOVIE_REVIEWS_CHUNKS_PATH )
    except OSError:
        pass

    texts_to_chunk_lists( texts_positive, texts_negative, MOVIE_REVIEWS_CHUNKS_PATH )
