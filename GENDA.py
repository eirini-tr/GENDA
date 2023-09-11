# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 11:59:22 2022

@author: troullinou
"""


################################ IMPORTS ################################
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from evaluation_performance import evaluation
from create_imbalanced_imaging_data import create_imbalanced_imaging_data
from create_imbalanced_HAR_dataset import create_imbalanced_har_data
from create_imbalanced_UCR_dataset import create_imbalanced_twoleadecg_data




############################## LOAD DATASETS ##############################
dataset = 'mnist'  # Select dataset: mnist - fmnist - har - twoleadecg

if dataset=='mnist' or dataset=='fmnist':
    train_data, train_labels, test_data, test_labels = create_imbalanced_imaging_data(dataset)
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], 1).astype('float32')
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], 1).astype('float32')
    # SCALING
    train_data = train_data / 255.
    test_data = test_data / 255.
    x_tr = np.reshape(train_data, [len(train_data), train_data.shape[1]*train_data.shape[2]])

    
elif dataset=='har':
    
    # Firstly download the HAR dataset from the UCI Machine Learning Repository: 
    # https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
    
    train_data, train_labels, test_data, test_labels = create_imbalanced_har_data()
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], 1).astype('float32')
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], 1).astype('float32')
    x_tr = np.reshape(train_data, [len(train_data), train_data.shape[1]*train_data.shape[2]])


elif dataset=='twoleadecg':
    
    # Firstly download the TwoLeadECG dataset from the UCR Time Series Classification Archive:
    # https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
    
    train_data, train_labels, test_data, test_labels = create_imbalanced_twoleadecg_data()
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1]).astype('float32')
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], 1).astype('float32')
    x_tr = np.reshape(train_data, [len(train_data), train_data.shape[1]])


n_classes = max(train_labels) + 1

channels = 1

################################ FIND NEIGHBORS ################################
nbrs_tr = NearestNeighbors(n_neighbors=3, algorithm='ball_tree', metric='euclidean').fit(x_tr)
distances_tr, indices_tr = nbrs_tr.kneighbors(x_tr)




def make_pairs(x, y, indices):
    """Creates a tuple containing image pairs with corresponding label.

    Arguments:
        x: List containing images, each index in this list corresponds to one image.
        y: List containing labels, each label with datatype of `int`.

    Returns:
        Two numpy arrays: pairs_of_samples, and their labels
    """


    pairs = []
    label_true = []
    
    
    for idx1 in range(len(x)):
        
        x1 = x[idx1]
        label1 = y[idx1]
        
        idx2 = indices[idx1,1]
        x2 = x[idx2]
        
        idx3 = indices[idx1,2]
        x3 = x[idx3]
        
        pairs += [[x1, x2, x3]]
        label_true +=[[label1]]
          
    
    return np.array(pairs), np.array(label_true).astype("float32")


pairs, label_tr = make_pairs(train_data, train_labels, indices_tr)
# Randomly shuffled
indx = np.random.permutation(len(label_tr))
pairs_train, label_true = pairs[indx], label_tr[indx]





########################### FIND THE EUCLIDEAN DISTANCE BETWEEN 2 VECTORS ###########################
def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:import random
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x,y = vects
       
    u = tf.random.normal(shape=[], mean=0, stddev=1, dtype=tf.float32)
          
    sum_ = tf.math.add(u*x,y*(1-u))
    

    return sum_



################################ BUILD THE ENCODER MODEL ################################
latentDim = 16 
def encoder_model():
       
    if dataset=='mnist' or dataset=='fmnist':
        
        input_ = layers.Input((train_data.shape[1], train_data.shape[2], channels))
        input_1 = layers.Input((train_data.shape[1], train_data.shape[2], channels))
        input_2 = layers.Input((train_data.shape[1], train_data.shape[2], channels))
      
        x = tf.keras.layers.BatchNormalization()(input_)
        
        x = layers.Conv2D(16, kernel_size = (4, 4), activation="tanh", padding='same')(x)
        x = layers.AveragePooling2D(pool_size=(2, 2))(x)
        
        x = layers.Conv2D(32, kernel_size = (4, 4), activation="tanh", padding='same')(x)
        x = layers.AveragePooling2D(pool_size=(2, 2))(x)
        
        x = layers.Conv2D(64, kernel_size = (4, 4), activation="tanh", padding='same')(x)
        x = layers.AveragePooling2D(pool_size=(2, 2))(x)
        
        x = layers.Conv2D(128, kernel_size = (4, 4), activation="tanh", padding='same')(x)
        x = layers.AveragePooling2D(pool_size=(2, 2))(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(latentDim, activation="tanh")(x)
        
    
    elif dataset=='har':
        
        input_ = layers.Input((train_data.shape[1], train_data.shape[2], channels))
        input_1 = layers.Input((train_data.shape[1], train_data.shape[2], channels))
        input_2 = layers.Input((train_data.shape[1], train_data.shape[2], channels))
        
        x = tf.keras.layers.BatchNormalization()(input_)
    
        x = layers.Conv2D(16, kernel_size = (2, 2), activation="tanh", padding='valid')(x)
        
        x = layers.Conv2D(32, kernel_size = (2, 2), activation="tanh", padding='valid')(x)
        x = layers.AveragePooling2D(pool_size=(2, 2), padding='valid')(x)
        
        x = layers.Conv2D(64, kernel_size = (2, 2), activation="tanh", padding='valid')(x)
        x = layers.AveragePooling2D(pool_size=(2, 2), padding='valid')(x)
        
        x = layers.Flatten()(x)
        
        x = tf.keras.layers.BatchNormalization()(x)
        x = layers.Dense(latentDim, activation="tanh")(x)
        
        
    elif dataset=='twoleadecg':
               
        input_ = layers.Input((train_data.shape[1], 1))
        input_1 = layers.Input((train_data.shape[1], 1))
        input_2 = layers.Input((train_data.shape[1], 1))
    
        x = tf.keras.layers.BatchNormalization()(input_)
    
    
        x = layers.Conv1D(16, kernel_size = 2, activation="tanh", padding='valid')(x)
    
        x = layers.Conv1D(32, kernel_size = 2, activation="tanh", padding='valid')(x)
        x = layers.AveragePooling1D(pool_size = 2, padding='valid')(x)
    
        x = layers.Conv1D(64, kernel_size = 2, activation="tanh", padding='valid')(x)
        x = layers.AveragePooling1D(pool_size = 2, padding='valid')(x)
    
    
        x = layers.Flatten()(x)
    
        x = tf.keras.layers.BatchNormalization()(x)
        x = layers.Dense(latentDim, activation="tanh")(x)
        
               
    embedding_network = keras.Model(input_, x)
           
    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)
   
    
    z = layers.Lambda(euclidean_distance)([tower_1,tower_2 ])
       
    
    encoder = keras.Model(inputs=[input_1,input_2], outputs=[z])


    return encoder

enc = encoder_model()



################################ BUILD THE DECODER MODEL ################################
input_decoder = (latentDim,)
def decoder(input_decoder):
    
    inputs = keras.Input(shape=input_decoder, name='input_layer')
    
    if dataset=='mnist' or dataset=='fmnist':            
        x = layers.Dense(7*7*128, name='dense_1')(inputs)
        x = layers.Reshape((7, 7, 128), name='Reshape_Layer')(x)
       
       
        # # Block-1
        x = layers.Conv2DTranspose(128, kernel_size = (3,3), strides= (1,1), padding='same',name='conv_transpose_1')(x)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.LeakyReLU(name='lrelu_1')(x)
      
        # Block-2
        x = layers.Conv2DTranspose(64, kernel_size = (3,3), strides= (2,2), padding='same', name='conv_transpose_2')(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.LeakyReLU(name='lrelu_2')(x)
        
        # Block-3
        x = layers.Conv2DTranspose(32, kernel_size = (3,3), strides= (2,2), padding='same', name='conv_transpose_3')(x)
        x = layers.BatchNormalization(name='bn_3')(x)
        x = layers.LeakyReLU(name='lrelu_3')(x)
    
        # Block-4
        outputs = layers.Conv2DTranspose(1, kernel_size = (3,3), strides= (1,1), padding='same', activation='sigmoid', name='conv_transpose_4')(x)    
    
    
    elif dataset=='har':
        
        x = layers.Dense(128*9*32, name='dense_1')(inputs)
        x = layers.Reshape((128, 9, 32), name='Reshape_Layer')(x)
      
      
        # Block-1
        x = layers.Conv2DTranspose(64, kernel_size = (2,2), strides= (1,1), padding='same', name='conv_transpose_2')(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.LeakyReLU(name='lrelu_2')(x)
        
        # Block-2
        x = layers.Conv2DTranspose(32, kernel_size = (2,2), strides= (1,1), padding='same', name='conv_transpose_3')(x)
        x = layers.BatchNormalization(name='bn_3')(x)
        x = layers.LeakyReLU(name='lrelu_3')(x)
        
         # Block-3
        x = layers.Conv2DTranspose(16, kernel_size = (2,2), strides= (1,1), padding='same', name='conv_transpose_4')(x)
        x = layers.BatchNormalization(name='bn_4')(x)
        x = layers.LeakyReLU(name='lrelu_4')(x)
        
        # Block-4
        outputs = layers.Conv2DTranspose(1, kernel_size = (2,2), strides= (1,1), padding='same', activation='sigmoid', name='conv_transpose_5')(x)
     
    
    elif dataset=='twoleadecg':
        
        dim = 16
        x = layers.Dense(train_data.shape[1]*dim, name='dense_1')(inputs)
        x = layers.Reshape((train_data.shape[1], 1, dim), name='Reshape_Layer')(x)
   
  
        # Block-1
        x = layers.Conv2DTranspose(64, kernel_size = (2,1), strides= (1,1), padding='same', name='conv_transpose_2')(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.LeakyReLU(name='lrelu_2')(x)
        
        # Block-2
        x = layers.Conv2DTranspose(32, kernel_size = (2,1), strides= (1,1), padding='same', name='conv_transpose_3')(x)
        x = layers.BatchNormalization(name='bn_3')(x)
        x = layers.LeakyReLU(name='lrelu_3')(x)
        
         # Block-3
        x = layers.Conv2DTranspose(16, kernel_size = (2,1), strides= (1,1), padding='same', name='conv_transpose_4')(x)
        x = layers.BatchNormalization(name='bn_4')(x)
        x = layers.LeakyReLU(name='lrelu_4')(x)
        
        # Block-4
        outputs = layers.Conv2DTranspose(1, kernel_size = (2,1), strides= (1,1), padding='same', activation='sigmoid', name='conv_transpose_5')(x)
    
    
    decoder = tf.keras.Model(inputs, outputs, name="Decoder")
        
        
    return decoder


dec = decoder(input_decoder)



############################ DEFINE LOSS, OPTIMIZER ############################
###################### LINK ENCODER & DECODER IN ONE MODEL #####################
GENDA_optimizer = keras.optimizers.Adam(learning_rate=0.0001)

# Link encoder & decoder in one model
if dataset=='mnist' or dataset=='fmnist' or dataset=='har':
    GENDA_input1 = tf.keras.layers.Input(shape=(train_data.shape[1], train_data.shape[2], channels), name="GENDA_input1")
    GENDA_input2 = tf.keras.layers.Input(shape=(train_data.shape[1], train_data.shape[2], channels), name="GENDA_input2")

elif dataset=='twoleadecg':
    GENDA_input1 = tf.keras.layers.Input(shape=(train_data.shape[1], 1), name="GENDA_input1")
    GENDA_input2 = tf.keras.layers.Input(shape=(train_data.shape[1], 1), name="GENDA_input2")
    

GENDA_encoder_output = enc([GENDA_input1, GENDA_input2])
GENDA_decoder_output = dec(GENDA_encoder_output)
GENDA = tf.keras.models.Model([GENDA_input1, GENDA_input2], GENDA_decoder_output, name="GENDA")



################################ APPLY GRADIENTS ################################
@tf.function
def train_step(data, label_true):


    with tf.GradientTape() as GENDA_tape:
    

        code = enc([data[:,0], data[:,1]], training=True)   
        generated_data = dec([code], training=True)        
               
        loss = tf.keras.losses.mean_squared_error(data[:,0], generated_data)
      
    
    GENDA_grads = GENDA_tape.gradient(loss, GENDA.trainable_weights)     
    GENDA_optimizer.apply_gradients(zip(GENDA_grads, GENDA.trainable_weights))
      
              
    return loss




########################### START TRAINING THE AUTOENCODER ########################
BATCH_SIZE = 16
epochs = 40

print('Start training the proposed autoencoder')    
for epoch in range(epochs):
    
    for i in range(0, len(pairs_train), BATCH_SIZE):
    
        train_step(pairs_train[i:i+BATCH_SIZE], label_true[i:i+BATCH_SIZE]) 

    
    print("Epoch %d" % (epoch))
    

   

    
######################## GENERATE NEW SAMPLES FOR EACH CLASS ########################
present_classes, counts = np.unique(train_labels, return_counts=True)
majority_count_idx = np.argmax(counts, axis=0)   
majority_label, majority_count = present_classes[majority_count_idx], counts[majority_count_idx]



idx = []
label_tr = np.squeeze(label_tr)
for i in range(0, n_classes):
    idx.append(np.where(label_tr==i))



#####1
idx1 = idx[1]
p = pairs[idx1]  
g1data = []  
g1labels = []
if majority_count != len(idx1[0]):
    
    for i in range( int(np.floor(majority_count / len(idx1[0]))) ):
        
        code = enc([p[:,0], p[:,1]], training=False)                
        generated_data = dec([code], training=False)
        g1data.append(generated_data)   

g1data = np.concatenate(g1data) if g1data!=[] else train_data[idx1]
g1labels = 1*np.ones((int(np.floor(majority_count / len(idx1[0])))*counts[1]))
   
  

##########################################################
# If dataset = 'twoleadecg': Data generation is completed 
##########################################################

if dataset=='mnist' or dataset=='fmnist' or dataset=='har':

    #####2
    idx2 = idx[2]
    p = pairs[idx2] 
    g2data = []  
    g2labels = []
    if majority_count != len(idx2[0]):
        
        for i in range( int(np.floor(majority_count / len(idx2[0]))) ):
            
            code = enc([p[:,0], p[:,1]], training=False)
            generated_data = dec([code], training=False)
            g2data.append(generated_data)   
    
    g2data = np.concatenate(g2data) if g2data!=[] else train_data[idx2]
    g2labels = 2*np.ones((int(np.floor(majority_count / len(idx2[0])))*counts[2]))
      
    
    
    #####3
    idx3 = idx[3]
    p = pairs[idx3]    
    g3data = []  
    g3labels = []
    if majority_count != len(idx3[0]):
        
        for i in range( int(np.floor(majority_count / len(idx3[0]))) ):
            
            code = enc([p[:,0], p[:,1]], training=False)           
            generated_data = dec([code], training=False)
            g3data.append(generated_data)   
            
    g3data = np.concatenate(g3data) if g3data!=[] else train_data[idx3]
    g3labels = 3*np.ones((int(np.floor(majority_count / len(idx3[0])))*counts[3]))
        
    
    
    #####4 
    idx4 = idx[4]
    p = pairs[idx4]    
    g4data = [] 
    g4labels = [] 
    if majority_count != len(idx4[0]):
        
        for i in range( int(np.floor(majority_count / len(idx4[0]))) ):
            
            code = enc([p[:,0], p[:,1]], training=False)           
            generated_data = dec([code], training=False)
            g4data.append(generated_data)   
        
    g4data = np.concatenate(g4data) if g4data!=[] else train_data[idx4]  
    g4labels = 4*np.ones((int(np.floor(majority_count / len(idx4[0])))*counts[4]))
           
    
    #####5 
    idx5 = idx[5]
    p = pairs[idx5]    
    g5data = [] 
    g5labels = [] 
    if majority_count != len(idx5[0]):
        
        for i in range( int(np.floor(majority_count / len(idx5[0]))) ):
            
            code = enc([p[:,0], p[:,1]], training=False)
            generated_data = dec([code], training=False)
            g5data.append(generated_data)   
        
    g5data = np.concatenate(g5data) if g5data!=[] else train_data[idx5]  
    g5labels = 5*np.ones((int(np.floor(majority_count / len(idx5[0])))*counts[5]))
          
    
    ###################################################
    # If dataset = 'har': Data generation is completed 
    ###################################################
    
    if dataset=='mnist' or dataset=='fnist':
    
        #####6
        idx6 = idx[6]
        p = pairs[idx6]    
        g6data = [] 
        g6labels = []
        if majority_count != len(idx6[0]):
            
            for i in range( int(np.floor(majority_count / len(idx6[0]))) ):
                
                code = enc([p[:,0], p[:,1]], training=False)                       
                generated_data = dec([code], training=False)
                g6data.append(generated_data)   
            
        g6data = np.concatenate(g6data) if g6data!=[] else train_data[idx6]  
        g6labels = 6*np.ones((int(np.floor(majority_count / len(idx6[0])))*counts[6]))
                
        
        #####7
        idx7 = idx[7] 
        p = pairs[idx7]    
        g7data = [] 
        g7labels = [] 
        if majority_count != len(idx7[0]):
            
            for i in range( int(np.floor(majority_count / len(idx7[0]))) ):
                
                code = enc([p[:,0], p[:,1]], training=False)                       
                generated_data = dec([code], training=False)
                g7data.append(generated_data)   
        
        g7data = np.concatenate(g7data) if g7data!=[] else train_data[idx7]  
        g7labels = 7*np.ones((int(np.floor(majority_count / len(idx7[0])))*counts[7]))
                
        
        #####8 
        idx8 = idx[8]
        p = pairs[idx8]    
        g8data = [] 
        g8labels = [] 
        if majority_count != len(idx8[0]):
            
            for i in range( int(np.floor(majority_count / len(idx8[0]))) ):
                
                code = enc([p[:,0], p[:,1]], training=False)                
                generated_data = dec([code], training=False)
                g8data.append(generated_data)   
            
        g8data = np.concatenate(g8data) if g8data!=[] else train_data[idx8]  
        g8labels = 8*np.ones((int(np.floor(majority_count / len(idx8[0])))*counts[8]))
                
        
        #####9
        idx9 = idx[9]
        p = pairs[idx9]    
        g9data = [] 
        g9labels = [] 
        if majority_count != len(idx9[0]):
            
            for i in range( int(np.floor(majority_count / len(idx9[0]))) ):
                
                code = enc([p[:,0], p[:,1]], training=False)                       
                generated_data = dec([code], training=False)
                g9data.append(generated_data)   
            
        g9data = np.concatenate(g9data) if g9data!=[] else train_data[idx9]  
        g9labels = 9*np.ones((int(np.floor(majority_count / len(idx9[0])))*counts[9]))
                

################################################################
# If dataset = 'mnist' or 'fmnist': Data generation is completed 
################################################################



########################### MERGE ORIGINAL & GENERATED DATA #######################
if dataset=='mnist' or dataset=='fmnist':            
    g_all = [g1data, g2data, g3data, g4data, g5data, g6data, g7data, g8data, g9data]
    g_all_new = [x for x in g_all if len(x)>0]
    g_all_new = np.concatenate(g_all_new)
    
    g_labels_all = [g1labels, g2labels, g3labels, g4labels, g5labels, g6labels, g7labels, g8labels, g9labels]
    g_labels_all_new = [x for x in g_labels_all if len(x)>0]
    g_labels_all_new = np.concatenate(g_labels_all_new)
    
elif dataset=='har':
    
    g_all = [g1data, g2data, g3data, g4data, g5data]
    g_all_new = [x for x in g_all if len(x)>0]
    g_all_new = np.concatenate(g_all_new)
    
    g_labels_all = [g1labels, g2labels, g3labels, g4labels, g5labels]
    g_labels_all_new = [x for x in g_labels_all if len(x)>0]
    g_labels_all_new = np.concatenate(g_labels_all_new)
    
elif dataset=='twoleadecg':
    
    g_all = [g1data]
    g_all_new = [x for x in g_all if len(x)>0]
    g_all_new = np.concatenate(g_all_new)
    g_all_new = np.squeeze(g_all_new)


    g_labels_all = [g1labels]
    g_labels_all_new = [x for x in g_labels_all if len(x)>0]
    g_labels_all_new = np.concatenate(g_labels_all_new)
    
all_data = np.vstack((train_data, g_all_new))
all_labels = np.hstack((train_labels, g_labels_all_new ))   



################################ EVALUATION ################################
acsa, f1_macro, precision  = evaluation(all_data, all_labels, test_data, test_labels, dataset, n_classes)



################ PLOT GENERATED IMAGES FOR MNIST & FASHION-MNIST ################
if dataset=='mnist' or dataset=='fmnist':
    
    label = 9      # select the class of the image that you want to plot (from 0 to 9)
    ind = np.where(label_true==label)[0]    
    
    pt = pairs_train[ind[39]]
    pt = np.expand_dims(pt, 0)
    code = enc([pt[:,0], pt[:,1]], training=False)
    generated_data = dec([code], training=False)
    
    
    plt.imshow((np.squeeze(pt[:,0,:,:])), cmap=plt.get_cmap('gray'))   # raw image
    plt.imshow(np.squeeze(pt[:,1,:,:]), cmap=plt.get_cmap('gray'))     # neighbour 1
    plt.imshow(np.squeeze(pt[:,2,:,:]), cmap=plt.get_cmap('gray'))     # neighbour 2
    plt.imshow(np.squeeze(generated_data), cmap=plt.get_cmap('gray'))  # generated image










