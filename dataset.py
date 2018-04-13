# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 22:18:37 2018

@author: Rizwan Yasin
"""
import utils
dir_path = 'D:/datasets'
def load_dataset(title):
    if title=='glass-identification':
        #Glass Identification Data Set [214,9, 7, N, N]
        #URL: https://archive.ics.uci.edu/ml/datasets/glass+identification
        X,y = utils.load_data(dir_path+'/glass/glass.data', 9, index_col=None)
    elif title=='soybean-large':
        #Soybeans (Large) Data Set [307, 35, 19, Y, C]
        #https://archive.ics.uci.edu/ml/datasets/Soybean+(Large)
        X,y = utils.load_data(dir_path+'/soybean/soybean-large.data',35, last=False)
        y = utils.encode_label(y)
    elif title=='primary-tumor':
        #Primary Tumor Data Set [339x17x22xY,N]
        #https://archive.ics.uci.edu/ml/datasets/primary+tumor
        X,y = utils.load_data(dir_path+'/primary-tumor/primary-tumor.data',17, last=False)
    elif title=='winequality-red':
        #7a. Wine Quality Red Data Set [1599,11, 10, N, N]
        #[accuracy is not upto 54% with red and white wine data]
        #URL: https://archive.ics.uci.edu/ml/datasets/Wine+Quality
        X,y = utils.load_data(dir_path+'/wine_quality/winequality-red.csv', 11, header='infer', sep=';', col_name=True, target=['quality'])
    else:
        print('No dataset found for loading, please check again...')
    return X, y
