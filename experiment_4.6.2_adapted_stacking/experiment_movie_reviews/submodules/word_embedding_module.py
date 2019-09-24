import numpy as np
from flair.data import Sentence
from flair.embeddings import WordEmbeddings

def load_word_embeddings():
    print("Loading fastText embeddings...")
    de_fastText_embeddings = WordEmbeddings('en')
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
