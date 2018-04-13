# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 22:18:46 2018

@author: dell
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


import dataset
import batch

data_sets = [
'glass-identification', #8. Glass Identification Data Set [214,9, 7, N, N]
'soybean-large', #11. Soybeans (Large) Data Set [307, 35, 19, Y, C]
'primary-tumor', #14. Primary Tumor Data Set [339x17x22xY,N]
'winequality-red', #15a. Wine Quality Red Data Set [1599,11, 10, N, N]
]
classifiers = ['None','GaussianNB', 'KNeighborsClassifier','SVC','LogisticRegression']
methods = ['schulze','kemeny_young']
base_estimator=None
for ds in data_sets:
    print('#######################[BEGIN {}]###################'.format(ds))
    print('Loading starts...')
    X,y = dataset.load_dataset(ds)
    #X = feature_selection(X,y,  method='PCA')
    print('Loading completes...')
    for classifier in classifiers:
        #if not os.path.exists(directory):
        #    os.makedirs(directory)
        #output = open('d:\\results\\results_iris_{}.txt'.format(n_estimators), 'a')
        if classifier != 'None':
            base_estimator = eval(classifier+'()')
        file_name = '{}_{}'.format(ds,classifier)
        batch.predict_bagging(X,y,file_name,base_estimator=base_estimator, methods=methods)
        print('Classifier : '+classifier)
    print('#####################[END {}]#####################'.format(ds))