import os
import glob
import pandas as pd

def collect_files_from_paths( paths ):
    texts = []
    for p in paths:
        with open( p, 'r') as f:
            try:
                content = f.read()
                texts.append( content )
            except Exception as e:
                pass
    return texts


if __name__ == "__main__":
    print()
    base_path = "C:/Users/Felix/Downloads/aclImdb_v1/aclImdb"

    path_negative_samples_train = os.path.join( base_path, "train/neg" )
    train_files_names_negative = glob.glob( path_negative_samples_train + "/*.txt" )
    negative_texts_train = collect_files_from_paths( train_files_names_negative )
    negative_labels_train = [0]*len(negative_texts_train)


    path_positive_samples_train = os.path.join( base_path, "train/pos" )
    train_files_names_positive = glob.glob( path_positive_samples_train + "/*.txt" )
    positive_texts_train = collect_files_from_paths( train_files_names_positive )
    positive_labels_train = [1]*len(positive_texts_train)


    path_negative_samples_test = os.path.join( base_path, "test/neg" )
    test_files_names_negative = glob.glob( path_negative_samples_test + "/*.txt" )
    negative_texts_test = collect_files_from_paths( test_files_names_negative )
    negative_labels_test = [0]*len(negative_texts_test)

    path_positive_samples_test = os.path.join( base_path, "test/pos" )
    test_files_names_positive = glob.glob( path_positive_samples_test + "/*.txt" )
    positive_texts_test = collect_files_from_paths( test_files_names_positive )
    positive_labels_test = [1]*len(positive_texts_test)

    all_texts = negative_texts_train + positive_texts_train + negative_texts_test + positive_texts_test
    all_labels = negative_labels_train + positive_labels_train + negative_labels_test + positive_labels_test

    df = pd.DataFrame( { "text":all_texts, "label":all_labels})
    df.to_csv( "movie_reviews.csv", sep=";", index=None )
