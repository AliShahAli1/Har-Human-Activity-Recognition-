#!/usr/bin/env python
# coding: utf-8

# # Human Activity Recognitioon

# ### Sprint-1

# ### Import Libraries

# In[1]:


import numpy as np 
import pandas as pd 
from scipy import signal
import matplotlib.pyplot as plt 
import math
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


# ### Dataset Visualization 
# #### 1. load the  dataset_1 with each activity
# #### 2. visualize the dataset using matplot the accelerometer and gyroscope values are used for visualization with ( wrist, Chest, Hip, Anckle)
# 

# In[2]:


#.....
#Visualze the Accelerometer and Gyroscope data of wrist position with all categories and Partiicipants
#....Participant--19
#....Activities--13
#....Sensors two (Accelerometer and Gyroscope)
#.....

def data_visulization():
    
    for b in range(1,20):
        for a in range(1,14):
            # read dataset file
            df = pd.read_csv('dataset_'+str(b)+'.txt', sep=',', header=None)
            if a==1:
                label='sittinng'
            elif a==2:
                label='Lying'
            elif a==3:
                label='Standing'
            elif a==4:
                label='Washing dishes'
            elif a==5:
                label='Vacuuming'
            elif a==6:
                label='Sweeping'
            elif a==7:
                label='Walking'
            elif a==8:
                label='Ascending stairs'
            elif a==9:
                label='Descending stairs'
            elif a==10:
                label='Treadmill running'
            elif a==11:
                label='Bicycling on ergometer (50W)'
            elif a==12:
                label='Bicycling on ergometer (100W)'
            else:
                label='Rope jumping'
            df_label = df[df[24] == a].values
            #Visualisation with Acceleroometer.....
            print("Participant_: ",b)
            plt.plot(df_label[:,0:3])
            print(label+":  Accelerometer ")
            plt.show()
            #Visualiisation  with Gyroscope
            print("Participant_: ",b)
            print(label+":  Gyroscope")
            plt.plot(df_label[:,5:6])
            plt.show()


# ###### Note:
#  we can use the hip, chest and ankle  data by changiing the df_label index
# #...

# ### Calling out the visualization method

# In[3]:


# data_visulization()


# ### Sprint-2
# 1. Noise Removal 
# 2. Visualization after applying low pass filter

# In[4]:


#'''
#For raw sensor data, it usually contains noise that arises from different sources, such as sensor mis-
#calibration, sensor errors, errors in sensor placement, or noisy environments. We could apply filter to remove noise of sensor data
#to smooth data. In this example code, Butterworth low-pass filter is applied. 
#'''
def noise_removing():
    for e in range(1,20):
        for a in range(1,14):
            df = pd.read_csv('dataset_'+str(e)+'.txt', sep=',', header=None)
            if a==1:
                label='sittinng'
            elif a==2:
                label='Lying'
            elif a==3:
                label='Standing'
            elif a==4:
                label='Washing dishes'
            elif a==5:
                label='Vacuuming'
            elif a==6:
                label='Sweeping'
            elif a==7:
                label='Walking'
            elif a==8:
                label='Ascending stairs'
            elif a==9:
                label='Descending stairs'
            elif a==10:
                label='Treadmill running'
            elif a==11:
                label='Bicycling on ergometer (50W)'
            elif a==12:
                label='Bicycling on ergometer (100W)'
            else:
                label='Rope jumping'
        # Butterworth low-pass filter. You could try different parameters and other filters. 
            b, d = signal.butter(4, 0.04, 'low', analog=False)
            df_label = df[df[24] == a].values
            for i in range(3):
                df_label[:,i] = signal.lfilter(b, d, df_label[:, i])
            plt.plot(df_label[:, 0:3])
            print("Participant_: ",b)
            print("Accelerometer Data",label)
            plt.show()


# ### Calling out the noise method

# In[5]:


# noise_removing()


# ## Sprint- 3
# 1. Feauture Engineering
# 2. Noise Removal
# 3. Splitting Training and Testing dataset

# In[6]:


#'''
#To build a human activity recognition system, we need to extract features from raw data and create feature dataset for training 
#machine learning models.

#Please create new functions to implement your own feature engineering. The function should output training and testing dataset.
#'''
def feature_engineering_Noise():
    training = np.empty(shape=(0, 49))
    testing = np.empty(shape=(0, 49))
    # deal with each dataset file
    for i in range(19):
        df = pd.read_csv('dataset_' + str(i + 1) + '.txt', sep=',', header=None)
        print('deal with dataset ' + str(i + 1))
        for c in range(1, 14):
            activity_data = df[df[24] == c].values
            b, a = signal.butter(4, 0.04, 'low', analog=False)
            for j in range(24):
                activity_data[:, j] = signal.lfilter(b, a, activity_data[:, j])
            
            datat_len = len(activity_data)
#             print(datat_len)
            training_len = math.floor(datat_len * 0.8)
            training_data = activity_data[:training_len, :]
            testing_data = activity_data[training_len:, :]

            # data segementation: for time series data, we need to segment the whole time series, and then extract features from each period of time
            # to represent the raw data. In this example code, we define each period of time contains 1000 data points. Each period of time contains 
            # different data points. You may consider overlap segmentation, which means consecutive two segmentation share a part of data points, to 
            # get more feature samples.
            training_sample_number = training_len // 1000 + 1
            testing_sample_number = (datat_len - training_len) // 1000 + 1
#             print("testingg sample number: ",testing_sample_number)
#             print("trainingg:",training_sample_number)


#Features Generation or extraction method.......
            
            for s in range(training_sample_number):
                if s < training_sample_number - 1:
                    sample_data = training_data[1000*s:1000*(s + 1), :]
                else:
                    sample_data = training_data[1000*s:, :]
                # in this example code, only three accelerometer data in wrist sensor is used to extract three simple features: min, max, and mean value in
                # a period of time. Finally we get 9 features and 1 label to construct feature dataset. You may consider all sensors' data and extract more
#                 print("samplle_data: ",sample_data)
                feature_sample = []
                for i in range(12):
                    feature_sample.append(np.min(sample_data[:, i]))
                    feature_sample.append(np.max(sample_data[:, i]))
                    feature_sample.append(np.mean(sample_data[:, i])) 
                    feature_sample.append(np.std(sample_data[:, i])) 
                feature_sample.append(sample_data[0, -1])
                feature_sample = np.array([feature_sample])
#                 print("featuure_samplle:",feature_sample.shape)
#                 print("training_data:",training.shape)
                
#                 print(feature_sample.shape)
                training = np.concatenate((training, feature_sample), axis=0)
            
            for s in range(testing_sample_number):
                if s < training_sample_number - 1:
                    sample_data = testing_data[1000*s:1000*(s + 1), :]
                else:
                    sample_data = testing_data[1000*s:, :]

                feature_sample = []
                for i in range(12):
                    feature_sample.append(np.min(sample_data[:, i]))
                    feature_sample.append(np.max(sample_data[:, i]))
                    feature_sample.append(np.mean(sample_data[:, i]))
                    feature_sample.append(np.std(sample_data[:, i]))
                
                feature_sample.append(sample_data[0, -1])
                feature_sample = np.array([feature_sample])
                testing = np.concatenate((testing, feature_sample), axis=0)
                

    df_training = pd.DataFrame(training)
    df_testing = pd.DataFrame(testing)
    df_training.to_csv('training_data_2.csv', index=None, header=None)
    df_testing.to_csv('testing_data_2.csv', index=None, header=None)


# In[7]:


feature_engineering_Noise()


# ### Machine Learning Modeling/ Classifiers
# 1. KNN (k nearest neighbor)
# 2. SVM (Support Vector Machine)

# In[10]:


#'''
#Please create new functions to fit your features and try other models.
#'''
def model_training_and_evaluation():
    
    df_training = pd.read_csv('training_data_2.csv', header=None)
    df_testing = pd.read_csv('testing_data_2.csv', header=None)

    y_train = df_training[48].values
    # Labels should start from 0 in sklearn
    y_train = y_train - 1
    df_training = df_training.drop([48], axis=1)
    X_train = df_training.values

    y_test = df_testing[48].values
    y_test = y_test - 1
    df_testing = df_testing.drop([48], axis=1)
    X_test = df_testing.values
    #............................Normalization........................
    
    
    # Feature normalization for improving the performance of machine learning models. In this example code, 
    # StandardScaler is used to scale original feature to be centered around zero. You could try other normalization methods.
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    #...........................KNN..................................
    
    # Build KNN classifier, in this example code
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Evaluation. when we train a machine learning model on training set, we should evaluate its performance on testing set.
    # We could evaluate the model by different metrics. Firstly, we could calculate the classification accuracy. In this example
    # code, when n_neighbors is set to 4, the accuracy achieves 0.757.
    y_pred = knn.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    # We could use confusion matrix to view the classification for each activity.
    print("Confusion Matrix\n")
    print(confusion_matrix(y_test, y_pred))
    
    #.............................Support Vector Machine...............................
    
    
    # Another machine learning model: svm. In this example code, we use gridsearch to find the optimial classifier
    # It will take a long time to find the optimal classifier.
    # the accuracy for SVM classifier with default parameters is 0.71, 
    # which is worse than KNN. The reason may be parameters of svm classifier are not optimal.  
    # Another reason may be we only use 9 features and they are not enough to build a good svm classifier. 
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1,1e-2, 1e-3, 1e-4],
                     'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 100]},
                    {'kernel': ['linear'], 'C': [1e-3, 1e-2, 1e-1, 1, 10, 100]}]
    acc_scorer = make_scorer(accuracy_score)
    grid_obj  = GridSearchCV(SVC(), tuned_parameters, cv=10, scoring=acc_scorer)
    grid_obj  = grid_obj .fit(X_train, y_train)
    clf = grid_obj.best_estimator_
    print('best clf:', clf)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print("Confusion Matrix\n")
    print(confusion_matrix(y_test, y_pred))

# print("# Tuning hyper-parameters for %s" % score)
# print()
# clf = GridSearchCV(SVC(), tuned_parameters, cv=10,
#                    scoring=score)
# clf.fit(x_train, y_train)

#........................................Main Method.................

if __name__ == '__main__':
    
    # data_visulization()
    # noise_removing()
    # feature_engineering_example()
    model_training_and_evaluation()


# In[ ]:





# In[ ]:




