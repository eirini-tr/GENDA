# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:53:01 2022

@author: troullinou
"""

import numpy as np
from scipy import stats


def create_imbalanced_twoleadecg_data():

          
    # Firstly download the TwoLeadECG dataset from the UCR Time Series Classification Archive:
    # https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
    # Change this path accordingly
    dataset = 'TwoLeadECG' 
    path = r'C:\Users\Datasets\Univariate_Timeseries\UCRArchive_2018/' +  dataset  

      
    def readucr(filename):
        data = np.loadtxt(filename, delimiter="\t")
        y = data[:, 0]
        x = data[:, 1:]
        return x, y.astype(int)
    
          
    x_train, y_train = readucr(path + "/" + dataset + "_TEST.tsv")
    x_train_norm = stats.zscore(x_train)
    y_train = y_train - 1
    
    x_test, y_test = readucr(path + "/" + dataset + "_TRAIN.tsv")
    test_data = stats.zscore(x_test)
    test_labels = y_test - 1
        
    
    # CREATE IMBALANCED TRAINING DATASET
    zero = np.where(y_train==0)[0]
    counts = 569
    ind = np.random.choice(zero,counts,replace=False)
    zero_data = x_train_norm[ind]
    zero_labels = y_train[ind]
    
    
    one = np.where(y_train==1)[0]
    counts = 40
    ind = np.random.choice(one,counts,replace=False)
    one_data = x_train_norm[ind]
    one_labels = y_train[ind]
    
    
    train_data = np.vstack((zero_data, one_data))
    train_labels = np.hstack(( np.squeeze(zero_labels), np.squeeze(one_labels) ))
        
        
    return train_data, train_labels, test_data, test_labels

        