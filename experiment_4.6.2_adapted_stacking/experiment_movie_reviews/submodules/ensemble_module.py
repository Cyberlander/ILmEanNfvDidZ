import re
import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

def get_meta_classifier():
    classifier = OneVsRestClassifier(LogisticRegression( C=1000000.0, penalty='l2', solver='sag' ))
    return classifier

def get_meta_classifier_to_bust_benchmark():
    classifier = OneVsRestClassifier(LogisticRegression( C=1000000.0, penalty='l2', solver='sag',max_iter=100000 ))
    return classifier

def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

def get_oof_with_probability(clf, x_train, y_train, x_test, ntrain, ntest):
    oof_train = np.zeros((ntrain,3))
    oof_test = np.zeros((ntest,3))
    oof_test_skf = np.empty((NFOLDS, ntest,3))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr.ravel())

        oof_train[test_index] = clf.predict_proba(x_te)
        oof_test_skf[i, :] = clf.predict_proba(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 3), oof_test.reshape(-1, 3)
