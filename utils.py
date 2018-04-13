# -*- coding: utf-8 -*-

from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn import metrics

def measures(y, predicted):
    measures = {
            'cohen_kappa_score':'cohen_kappa_score',
            'accuracy_score':'accuracy_score',
            'confusion_matrix':'confusion_matrix',
            'r2_score':'r2_score', 
            'hamming_loss':'hamming_loss',
            'jaccard_similarity_score':'jaccard_similarity_score',
            'zero_one_loss':'zero_one_loss',
            'precision_recall_fscore_support':'precision_recall_fscore_support',
            'f1_score':['micro','macro','weighted'],
            'precision_score':['micro','macro','weighted'],
            'recall_score':['micro','macro','weighted']
            }
    metrics_result = {}
    for measure in measures:
        scorer = eval('metrics.'+measure)
        if type(measures[measure]) == type(''):
            metrics_result[measure] = scorer(y, predicted)
        else:
            for arg  in measures[measure]:
                metrics_result[measure+'-'+arg] = scorer(y, predicted, average=arg)
    #print(metrics_result)
    return metrics_result

def feature_selection(X,y, method='PCA'):
    print('Before {} Feature Selection X: {}'.format(method, X.shape))
    if method == 'LSVC':
        from sklearn.svm import LinearSVC
        from sklearn.feature_selection import SelectFromModel
        X = SelectFromModel(LinearSVC().fit(X, y), prefit=True).transform(X)
    elif method == 'TSNE':
        from sklearn.manifold import TSNE
        X = TSNE(n_components=2, init='pca').fit_transform(X)
    elif method == 'LDA':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        X = LinearDiscriminantAnalysis(n_components=2).fit(X, y).transform(X)
    elif method == 'ETC':
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.feature_selection import SelectFromModel
        X = SelectFromModel(ExtraTreesClassifier(random_state=0).fit(X, y), prefit=True).transform(X)
    elif method == 'RFE':
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression
        X = RFE(LogisticRegression()).fit(X,y)
    elif method == 'PCA':
        from sklearn.decomposition import PCA
        X = PCA(n_components=2).fit_transform(X)
    print('After {} Feature Selection X: {}'.format(method, X.shape))
    return X

import pickle

def save_obj(dictionary,File):
    with open(File, "wb") as myFile:
        pickle.dump(dictionary, myFile)
        myFile.close()

def load_obj(File):
    with open(File, "rb") as myFile:
        dict = pickle.load(myFile)
        myFile.close()
        return dict

def load_data(path, dim, sep=',', col_name=False, last=True, target=['class'], header=None, index_col=False):
    if col_name==False :
        #target = ['class']
        if last == True:
            names = ['S{}'.format(i) for i in range(0,dim)]+target
        else:
            names = target+['S{}'.format(i) for i in range(0,dim)]
        data = pd.read_csv(path, sep=sep, names=names, header=header, index_col=index_col)
    else :
        data = pd.read_csv(path, sep=sep, header=header, index_col=index_col)
    #data.head()
    print('Classwise Distribution:\n{}'.format(data.groupby(target).size()))
    print('Missing Values {} : {} '.format(data.isnull().values.any(), data.isnull().sum().sum()))
    if data.isnull().values.any()==True:
        data[list(set(data)-set(target))]=data.groupby(target).transform(lambda x: x.fillna(-9999999999))
    
    if last == True:
        X, y = data.values[:,0:dim], data.values[:,dim]
    else:
        X, y = data.values[:,1:dim+1], data.values[:,0]
    return X,y
#-----------------------------[train and test classifier]----------------------------
from sklearn import model_selection
def tts_estimator(estimator, X, y, test_size=0.3, random_state=10):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=random_state)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    return measures(y_test, y_pred)
#---------------------------------------------------------
#------------------------------[k-fold cross validation]---------------------------
def kfcv_estimator(estimator, X, y, n_splits=10, random_state=10):
    #print('k-fold cross validation:\n')
    kfold = model_selection.KFold(n_splits=n_splits, random_state=random_state)
    predicted = model_selection.cross_val_predict(estimator, X, y, cv=kfold)
    return measures(y, predicted)
#---------------------------------------------------------
from sklearn import preprocessing
def encode_label(y):
    le = preprocessing.LabelEncoder()
    le.fit(y)
    return le.transform(y)
