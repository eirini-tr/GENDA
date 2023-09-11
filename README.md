# A Generative Neighborhood-based Deep Autoencoder for Robust Imbalanced Classification

This repository contains the official implementation of the proposed method GENDA described in the "A Generative Neighborhood-based Deep Autoencoder for Robust Imbalanced Classification".

[Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10054417&tag=1)

# Abstract

Deep learning models perform remarkably well on many classification tasks recently. The superior performance of deep neural networks relies on the large number of training data, which at the same time must have an equal class
distribution in order to be efficient. However, in most real-world applications the labeled data may be limited with high imbalance ratios among the classes and thus the learning process of most classification algorithms is adversely affected resulting to unstable predictions and low performance. Three main categories of approaches address the problem of imbalanced learning, i.e. data level, algorithmic level and hybrid methods, which combine the two aforementioned approaches. Data generative methods are typically based on Generative Adversarial Networks, which require significant amounts of data, while model level methods entail extensive domain expert knowledge to craft the learning objectives, thereby being less accessible for users without such knowledge. Moreover, the vast majority of these approaches is designed and applied to imaging applications, less to time series, and extremely rarely to both of them. To address the above issues, we introduce GENDA, a generative neighborhood-based deep autoencoder, which is simple yet effective in its design and can be successfully applied to both image and time series data. GENDA is based on learning latent representations that rely on the neighboring embedding space of the samples. Extensive experiments, conducted on a variety of widely-used real datasets demonstrate the efficacy of the proposed method.

# Requirements

The code was tested on Tensorflow (2.2.0) and Keras (2.3.0), in Python (3.6)

# Contents

**create_imbalanced_imaging_data:** Function that creates an imbalanced training set of Mnist and Fashion-Mnist datasets and returns both the imbalanced training set and a testing set.

**create_imbalanced_HAR_dataset:** Function that creates an imbalanced training set of the [HAR dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) and returns both the imbalanced training set and a testing set.

**create_imbalanced_UCR_dataset:** Function that creates an imbalanced training set of the [TwoLeadECG dataset](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) and returns both the imbalanced training set and a testing set.

**GENDA:** Performs the training of the proposed method GENDA and generates new artificial samples.

**evaluation_performance:** Function that evaluates the proposed method GENDA. It returns the average class specific accuracy (acsa), the F1-score and the precision of a CNN-classifier trained on a balanced training set consisting of the real and the generated data produced by GENDA. 

# Citation

E. Troullinou, G. Tsagkatakis, A. Losonczy, P. Poirazi and P. Tsakalides, "A Generative Neighborhood-based Deep Autoencoder for Robust Imbalanced Classification," in IEEE Transactions on Artificial Intelligence, doi: 10.1109/TAI.2023.3249685
