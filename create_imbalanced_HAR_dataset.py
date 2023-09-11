# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 14:51:37 2022

@author: troullinou
"""



from numpy import dstack
import numpy as np
from pandas import read_csv


def create_imbalanced_har_data():
 
    
    # Download the HAR dataset from the UCI Machine Learning Repository: 
    # https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
    # Change the path accordingly
    path_dir = r'C:/Users/Datasets/Univariate_Timeseries'

        
     
    # load a single file as a numpy array
    def load_file(filepath):
    	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    	return dataframe.values
     
    # load a list of files, such as x, y, z data for a given variable
    def load_group(filenames, prefix=''):
    	loaded = list()
    	for name in filenames:
    		data = load_file(prefix + name)
    		loaded.append(data)
    	# stack group so that features are the 3rd dimension
    	loaded = dstack(loaded)
    	return loaded
     
    # load a dataset group, such as train or test
    def load_dataset(path_dir, group, prefix=''):
    	filepath = path_dir + prefix + group + '/Inertial Signals/'
    	# load all 9 files as a single array
    	filenames = list()
    	# total acceleration
    	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
    	# body acceleration
    	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
    	# body gyroscope
    	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
    	# load input data
    	X = load_group(filenames, filepath)
    	# load class output
    	y = load_file(path_dir + prefix + group + '/y_'+group+'.txt')
    	return X, y
     
    
    # load all train
    trainX, trainy = load_dataset(path_dir, 'train', '/HARDataset/')
    trainy = trainy - 1
    # print(trainX.shape, trainy.shape)
    
    # load all test
    testX, testy = load_dataset(path_dir, 'test', '/HARDataset/')
    testy = testy - 1
    # print(testX.shape, testy.shape)
       
     
    # CREATE IMBALANCED TRAINING DATASET
    zero = np.where(trainy==0)[0]
    counts = 1226
    ind = np.random.choice(zero,counts,replace=False)
    zero_data = trainX[ind]
    zero_labels = trainy[ind]
    
    
    one = np.where(trainy==1)[0]
    counts = 800
    ind = np.random.choice(one,counts,replace=False)
    one_data = trainX[ind]
    one_labels = trainy[ind]
    
    
    two = np.where(trainy==2)[0]
    counts = 500
    ind = np.random.choice(two,counts,replace=False)
    two_data = trainX[ind]
    two_labels = trainy[ind]
    
    
    three = np.where(trainy==3)[0]
    counts = 300
    ind = np.random.choice(three,counts,replace=False)
    three_data = trainX[ind]
    three_labels = trainy[ind]
    
    
    four = np.where(trainy==4)[0]
    counts = 100
    ind = np.random.choice(four,counts,replace=False)
    four_data = trainX[ind]
    four_labels = trainy[ind]
       
    
    five = np.where(trainy==5)[0]
    counts = 40
    ind = np.random.choice(five,counts,replace=False)
    five_data = trainX[ind]
    five_labels = trainy[ind]

           
    train_data = np.vstack((zero_data, one_data, two_data, three_data, four_data, five_data))
    train_labels = np.hstack((np.squeeze(zero_labels), np.squeeze(one_labels), np.squeeze(two_labels), np.squeeze(three_labels), np.squeeze(four_labels), np.squeeze(five_labels)))
    
    
    return train_data, train_labels, testX, testy
