import pandas as pd

facebook_label_dict = {
    "csa":0,
    "dsa":1,
    "none":2
}
def load_facebook_data_from_csv( database_path ):
    df = pd.read_csv( database_path, header=0, delimiter=";", encoding="utf-8" )
    df = df.dropna()
    source_sentences = df['text'].tolist()
    label = df['label'].tolist()
    label_numeric = df['label'].map( facebook_label_dict )
    return source_sentences, label_numeric

def load_data_from_chunks( chunks_path, chunk_len ):
    all_source_sentences = []
    all_labels = []
    for i in range( chunk_len ):
        current_path = chunks_path.format( i )
        try:
            df = pd.read_csv( current_path, header=0, delimiter=";", encoding="utf-8" )
            df = df.dropna()
            all_source_sentences.append( df['text'].tolist() )
            all_labels.append( df['sentiment_numeric'].tolist() )
        except:
            print("Could not open the file {}".format( current_path ) )
    return all_source_sentences, all_labels
