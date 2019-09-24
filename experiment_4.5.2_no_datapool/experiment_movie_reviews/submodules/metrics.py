from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_metrics( y_true, y_to_pred ):
    acc = accuracy_score( y_true, y_to_pred )
    pre = precision_score( y_true, y_to_pred )
    rec = recall_score( y_true, y_to_pred )
    f1 = f1_score( y_true, y_to_pred )
    return acc, pre, rec, f1
