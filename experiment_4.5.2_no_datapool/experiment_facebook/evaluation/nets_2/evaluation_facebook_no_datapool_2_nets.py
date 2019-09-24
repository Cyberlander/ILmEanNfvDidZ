import pandas as pd
import numpy as np

def get_mean_value_from_dataframes( dataframes, column_name ):
    values_complete = []
    verlauf_matrix = np.zeros((5,50))
    for i,frame in enumerate(dataframes):
        values = frame[column_name].tolist()
        verlauf_matrix[i] = values

    mean_verlauf = np.mean(verlauf_matrix,axis=0)
    return np.mean(mean_verlauf)

    #return values_complete

if __name__ ==  "__main__":
    print()
    df_verlauf_1 = pd.read_csv("no_datapool_verlauf_facebook_2_nets_next_timestep_1.csv",delimiter=";")
    df_verlauf_2 = pd.read_csv("no_datapool_verlauf_facebook_2_nets_next_timestep_2.csv",delimiter=";")
    df_verlauf_3 = pd.read_csv("no_datapool_verlauf_facebook_2_nets_next_timestep_3.csv",delimiter=";")
    df_verlauf_4 = pd.read_csv("no_datapool_verlauf_facebook_2_nets_next_timestep_4.csv",delimiter=";")
    df_verlauf_5 = pd.read_csv("no_datapool_verlauf_facebook_2_nets_next_timestep_5.csv",delimiter=";")



    dataframes = [df_verlauf_1,df_verlauf_2,df_verlauf_3,df_verlauf_4,df_verlauf_5]

    #accuracy;precision_class_0;recall_class_0;f1_class_0;precision_class_1;recall_class_1;f1_class_1;precision_class_2;recall_class_2;f1_class_2;precision_class_3;recall_class_3;f1_class_3

    MEAN_ACC = get_mean_value_from_dataframes( dataframes, 'accuracy' )
    MEAN_PRE_1 = get_mean_value_from_dataframes( dataframes, 'precision_class_0' )
    MEAN_REC_1 = get_mean_value_from_dataframes( dataframes, 'recall_class_0' )
    MEAN_F1_1 = get_mean_value_from_dataframes( dataframes, 'f1_class_0' )
    MEAN_PRE_2 = get_mean_value_from_dataframes( dataframes, 'precision_class_1' )
    MEAN_REC_2 = get_mean_value_from_dataframes( dataframes, 'recall_class_1' )
    MEAN_F1_2 = get_mean_value_from_dataframes( dataframes, 'f1_class_1' )
    MEAN_PRE_3 = get_mean_value_from_dataframes( dataframes, 'precision_class_2' )
    MEAN_REC_3 = get_mean_value_from_dataframes( dataframes, 'recall_class_2' )
    MEAN_F1_3 = get_mean_value_from_dataframes( dataframes, 'f1_class_2' )

    print( 'accuracy: ', MEAN_ACC  )
    print( 'precision_class_0: ', MEAN_PRE_1  )
    print('recall_class_0: ', MEAN_REC_1 )
    print( 'f1_class_0: ', MEAN_F1_1   )
    print( 'precision_class_1: ', MEAN_PRE_2   )
    print('recall_class_1: ', MEAN_REC_2   )
    print( 'f1_class_1: ',MEAN_F1_2   )
    print( 'precision_class_2:', MEAN_PRE_3  )
    print( 'recall_class_2: ', MEAN_REC_3 )
    print( 'f1_class_2: ',MEAN_F1_3   )
