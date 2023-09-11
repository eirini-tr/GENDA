# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 15:41:08 2021

@author: troullinou
"""


import tensorflow as tf
import numpy as np
    

def create_imbalanced_imaging_data(dataset):
    
      
    if dataset=='mnist':
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    else:
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    
      
    
    # CREATE IMBALANCED TRAINING DATASET
    zero = np.where(train_labels==0)[0]
    counts = 4000
    ind = np.random.choice(zero,counts,replace=False)
    zero_data = train_images[ind]
    zero_labels = train_labels[ind]
    
    
    one = np.where(train_labels==1)[0]
    counts = 2000
    ind = np.random.choice(one,counts,replace=False)
    one_data = train_images[ind]
    one_labels = train_labels[ind]
    
    
    two = np.where(train_labels==2)[0]
    counts = 1000
    ind = np.random.choice(two,counts,replace=False)
    two_data = train_images[ind]
    two_labels = train_labels[ind]
    
    
    three = np.where(train_labels==3)[0]
    counts = 750
    ind = np.random.choice(three,counts,replace=False)
    three_data = train_images[ind]
    three_labels = train_labels[ind]
    
    
    four = np.where(train_labels==4)[0]
    counts = 500
    ind = np.random.choice(four,counts,replace=False)
    four_data = train_images[ind]
    four_labels = train_labels[ind]
    
    
    five = np.where(train_labels==5)[0]
    counts = 350
    ind = np.random.choice(five,counts,replace=False)
    five_data = train_images[ind]
    five_labels = train_labels[ind]
    
    
    six = np.where(train_labels==6)[0]
    counts = 200
    ind = np.random.choice(six,counts,replace=False)
    six_data = train_images[ind]
    six_labels = train_labels[ind]
    
    
    seven = np.where(train_labels==7)[0]
    counts = 100
    ind = np.random.choice(seven,counts,replace=False)
    seven_data = train_images[ind]
    seven_labels = train_labels[ind]
    
    
    eight = np.where(train_labels==8)[0]
    counts = 60
    ind = np.random.choice(eight,counts,replace=False)
    eight_data = train_images[ind]
    eight_labels = train_labels[ind]
    
    
    nine = np.where(train_labels==9)[0]
    counts = 40
    ind = np.random.choice(nine,counts,replace=False)
    nine_data = train_images[ind]
    nine_labels = train_labels[ind]
    
    
    train_data = np.vstack((zero_data, one_data, two_data, three_data, four_data, five_data, six_data, seven_data, eight_data, nine_data))
    train_labels = np.hstack((zero_labels, one_labels, two_labels, three_labels, four_labels, five_labels, six_labels, seven_labels, eight_labels, nine_labels))
    
    
    
    return train_data, train_labels, test_images, test_labels
    
    
