from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics_for_each_class(y_test_meta, meta_predictions ):
    n_classes = 3
    y_test_meta_binarized = label_binarize(y_test_meta, classes=[0, 1, 2])
    meta_predictions_binarized = label_binarize(meta_predictions, classes=[0, 1, 2])
    #accuracies = []
    precisions = []
    recalls = []
    f1s = []
    print()
    print(y_test_meta_binarized)
    print( meta_predictions_binarized)
    for i in range(n_classes):
        pre = precision_score( y_test_meta_binarized[:, i], meta_predictions_binarized[:, i] )
        rec = recall_score( y_test_meta_binarized[:, i], meta_predictions_binarized[:, i] )
        f1 = f1_score( y_test_meta_binarized[:, i], meta_predictions_binarized[:, i] )
        precisions.append(pre)
        recalls.append(rec)
        f1s.append(f1)
        print("Precision class {}: {}".format(i,pre))
        print("Recall class {}: {}".format(i,rec))
        print("F1 class {}: {}".format(i,f1))
    return precisions, recalls, f1s
