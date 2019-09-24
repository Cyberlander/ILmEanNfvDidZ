import pandas as pd
from submodules import data_io

def dataset_to_dataframe( text, label, file_name ):
    df = pd.DataFrame( { 'text' : text,
                         'sentiment_numeric':label } )
    df.to_csv( file_name, sep=";", index=None )


if __name__ =="__main__":
    CHUNKS_PATH = "chunks/yahoo_data_sentiment_classification_chunk_{}.csv"
    CHUNK_LEN = 5
    all_source_sentences, all_labels = data_io.load_data_from_chunks( CHUNKS_PATH, CHUNK_LEN )

    chunk1_sentences = all_source_sentences[0]
    chunk1_labels = all_labels[0]

    chunk2_sentences = all_source_sentences[1]
    chunk2_labels = all_labels[1]

    chunk3_sentences = all_source_sentences[2]
    chunk3_labels = all_labels[2]

    chunk4_sentences = all_source_sentences[3]
    chunk4_labels = all_labels[3]

    chunk5_sentences = all_source_sentences[4]
    chunk5_labels = all_labels[4]

    dataset_to_dataframe( chunk1_sentences, chunk1_labels, 'yahoo_growing_dataset/yahoo_growing_dataset_1.csv' )

    vergleichsnetz_zweites_trainingsset_sentences= chunk1_sentences + chunk2_sentences
    vergleichsnetz_zweites_trainingsset_labels = chunk1_labels + chunk2_labels
    dataset_to_dataframe( vergleichsnetz_zweites_trainingsset_sentences, vergleichsnetz_zweites_trainingsset_labels, 'yahoo_growing_dataset/yahoo_growing_dataset_2.csv' )


    vergleichsnetz_drittes_trainingsset_sentences = chunk1_sentences + chunk2_sentences + chunk3_sentences
    vergleichsnetz_drittes_trainingsset_labels = chunk1_labels + chunk2_labels + chunk3_labels
    dataset_to_dataframe( vergleichsnetz_drittes_trainingsset_sentences, vergleichsnetz_drittes_trainingsset_labels, 'yahoo_growing_dataset/yahoo_growing_dataset_3.csv' )


    vergleichsnetz_viertes_trainingsset_sentences = chunk1_sentences + chunk2_sentences + chunk3_sentences + chunk4_sentences
    vergleichsnetz_viertes_trainingsset_labels = chunk1_labels + chunk2_labels + chunk3_labels + chunk4_labels
    dataset_to_dataframe( vergleichsnetz_viertes_trainingsset_sentences, vergleichsnetz_viertes_trainingsset_labels, 'yahoo_growing_dataset/yahoo_growing_dataset_4.csv' )


    vergleichsnetz_fuenftes_trainingsset_sentences = chunk1_sentences + chunk2_sentences + chunk3_sentences + chunk4_sentences + chunk5_sentences
    vergleichsnetz_fuenftes_trainingsset_labels = chunk1_labels + chunk2_labels + chunk3_labels + chunk4_labels + chunk5_labels
    dataset_to_dataframe( vergleichsnetz_fuenftes_trainingsset_sentences, vergleichsnetz_fuenftes_trainingsset_labels, 'yahoo_growing_dataset/yahoo_growing_dataset_5.csv' )
