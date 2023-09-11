# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 16:47:34 2022

@author: troullinou
"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import confusion_matrix



def evaluation(all_data, all_labels, test_data, test_labels, dataset, n_classes):
    
    
    c_optimizer = keras.optimizers.Adam(lr=0.0001)   
        
    if dataset=='mnist' or dataset=='fmnist' or dataset=='har':
             
        channels = 1
        
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same',
                                         input_shape=[all_data.shape[1], all_data.shape[2], channels]))
        
        
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())        
        
        model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Conv2D(16, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Conv2D(8, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        
        model.add(layers.Flatten())
        model.add(layers.Dense(n_classes, activation='softmax'))
        
        model.compile(optimizer=c_optimizer,
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy'])
        
        
        if all_data.ndim==3: all_data = np.expand_dims(all_data,axis=-1) 
          
        model.fit(all_data, all_labels, epochs=50, shuffle=True)
        
    
    elif dataset=='twoleadecg':
                  
        model = tf.keras.Sequential()
        model.add(layers.Conv1D(128, 5, strides=2, padding='same',
                                         input_shape=[all_data.shape[1],1]))
        
        
        model.add(layers.Conv1D(64, 5, strides=2, padding='same'))
        model.add(layers.LeakyReLU())
                
        model.add(layers.Conv1D(32, 5, strides=2, padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Conv1D(16, 5, strides=1, padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Conv1D(8, 5, strides=2, padding='same'))
        model.add(layers.LeakyReLU())
        
        model.add(layers.Flatten())
        model.add(layers.Dense(n_classes, activation='softmax'))
        
        model.compile(optimizer=c_optimizer,
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy'])
        
        
        if all_data.ndim==2: all_data = np.expand_dims(all_data,axis=-1)
        
        print('Start training the classifier:')
          
        model.fit(all_data, all_labels, epochs=80, shuffle=True)
        
       

    """
    :return: acsa, pre, rec, f1_ma

    """
    
    test_predictions = model.predict_classes(test_data)
    
    test_labels, test_predictions = test_labels.astype(np.int8), test_predictions.astype(np.int8)

    cnf_matrix = confusion_matrix(test_labels, test_predictions)
    
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)

    acsa = TP / cnf_matrix.sum(axis=1)
    precision = (TP / (TP + FP + np.finfo(float).eps))
    recall = TP / (TP + FN + np.finfo(float).eps)
    f1_macro = (2 * precision * recall / (precision + recall + np.finfo(float).eps))

    return acsa.mean(), f1_macro.mean(), precision.mean()


