import re
import pandas as pd

label_dict = {
    'negative':0,
    'neutral':1,
    'mixed':2,
    'positive':3
}

def read_data( path, delimiter ):
    df = pd.read_csv( path, delimiter=delimiter, encoding='utf-8')
    return df

def data_chunking_sentiment( df_simple, chunk_count ):
    dfs_chunks = [None]*chunk_count
    df_negative = df_simple.loc[ df_simple['sentiment']=='negative' ]
    negative_comments = df_negative['text'].tolist()

    df_neutral = df_simple.loc[ df_simple['sentiment']=='neutral' ]
    neutral_comments = df_neutral['text'].tolist()

    df_mixed = df_simple.loc[ df_simple['sentiment']=='mixed' ]
    mixed_comments = df_mixed['text'].tolist()

    df_positive = df_simple.loc[ df_simple['sentiment']=='positive' ]
    positive_comments = df_positive['text'].tolist()

    chunk_size_negative = int(len(negative_comments)/chunk_count)
    chunk_size_neutral = int(len(neutral_comments)/chunk_count)
    chunk_size_mixed = int(len(mixed_comments)/chunk_count)
    chunk_size_positive = int(len(positive_comments)/chunk_count)

    chunks_negative_growing_dataset = []
    chunks_neutral_growing_dataset = []
    chunks_mixed_growing_dataset = []
    chunks_positive_growing_dataset = []
    chunks_sentiments_growing_dataset = []
    for i in range( chunk_count ):
        chunk_negative = negative_comments[i*chunk_size_negative:(i+1)*chunk_size_negative]
        chunk_neutral = neutral_comments[i*chunk_size_neutral:(i+1)*chunk_size_neutral]
        chunk_mixed = mixed_comments[i*chunk_size_mixed:(i+1)*chunk_size_mixed]
        chunk_positive = positive_comments[i*chunk_size_positive:(i+1)*chunk_size_positive]

        chunks_negative_growing_dataset.extend( chunk_negative )
        chunks_neutral_growing_dataset.extend( chunk_neutral )
        chunks_mixed_growing_dataset.extend( chunk_mixed )
        chunks_positive_growing_dataset.extend( chunk_positive )

        chunk_texts = chunk_negative + chunk_neutral + chunk_mixed + chunk_positive

        chunk_sentiments = len(chunk_negative)*['negative'] + len(chunk_neutral)*['neutral'] + len(chunk_mixed)*['mixed'] + len(chunk_positive)*['positive']


        chunk_df = pd.DataFrame({ 'text':chunk_texts, 'sentiment':chunk_sentiments })
        chunk_df['sentiment_numeric'] = chunk_df['sentiment'].map( label_dict )
        chunk_df.to_csv( "chunks/yahoo_data_chunk_{}.csv".format( i ), sep=";", index=None )



        chunk_texts_growing_dataset = chunks_negative_growing_dataset + chunks_neutral_growing_dataset + chunks_mixed_growing_dataset + chunks_positive_growing_dataset
        chunks_sentiments_growing_dataset.extend( chunk_sentiments )
        growing_dataset_df = pd.DataFrame({ 'text':chunk_texts_growing_dataset, 'sentiment':chunks_sentiments_growing_dataset })
        growing_dataset_df.to_csv( "growing_dataset/yahoo_data_growing_dataset_{}.csv".format( i ), sep=";", index=None )


def create_simple_df( comments, labels, label_name, filename, label_mapping  ):
    comments = [ re.sub(";",",",text) for text in comments ]
    df = pd.DataFrame( {'text':comments} )
    df[label_name] = labels
    label_name_numeric = label_name + "_numeric"
    df[label_name_numeric] = df[label_name].map( label_dict )
    df.to_csv( filename, sep=';', index=None )
    return df

def show_class_distribution( df ):
    #df_group_count = df.groupby('sentiment').count()
    series_count = df['sentiment'].value_counts()
    print(series_count)

if __name__ == "__main__":
    data_path = 'ydata-ynacc-v1_0_expert_annotations.tsv'
    delimiter = "\t"
    CHUNK_COUNT = 5
    CHUNK_COUNT = 10
    df = read_data( data_path, delimiter )
    #print( df.columns.values )

    # negative, mixed, neutral, positive, nan
    #print( df['sentiment'].unique()) #

    # tone - Controversial, Sarcastic, Informative

    headlines = df['headline'].tolist()
    tone_labels = df['tone'].tolist()

    comments = df['text'].tolist()
    sentiment_labels = df['sentiment'].tolist()

    show_class_distribution( df )

    df_sentiment = create_simple_df( comments, sentiment_labels,'sentiment',"yahoo_data_sentiment_classification.csv",label_dict )

    data_chunking_sentiment( df_sentiment, CHUNK_COUNT )
    #df_tone = create_simple_df( comments, tone_labels, )
    #print(comments[idx])
    #print(labels[idx])
