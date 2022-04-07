# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 19:48:00 2022

@author: jvillanuev29
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from Image_recog import load_data


#Read metadata data
raw_test_data = pd.read_csv('chestxray_corona_test_metadata.csv')
raw_test_data.groupby('Final_Label').size()
#Prepare indexes
covid_test = raw_test_data.loc[raw_test_data['Final_Label']=='Pnemonia_Virus_COVID-19',['X_ray_image_name']]
covid_test.reset_index(level=None,inplace=True,drop=True)
noncovid_test = raw_test_data.loc[raw_test_data['Final_Label']!='Pnemonia_Virus_COVID-19',['X_ray_image_name']]
noncovid_test.reset_index(level=None,inplace=True,drop=True)
#get images
X_test_covid = load_data('Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/',covid_test['X_ray_image_name'])
X_test_noncovid = load_data('Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/',noncovid_test['X_ray_image_name'])
X_test_final = np.concatenate((X_test_noncovid, X_test_covid),axis=0)

#Load train data and get onehhot columns
train_data = pd.read_csv('chestxray_corona_train_metadata.csv')
from sklearn.preprocessing import LabelBinarizer as LB
y_train = train_data['Final_Label']
lb = LB()
y_train_onehot = pd.DataFrame(lb.fit_transform(y_train))
y_train_onehot.columns = pd.get_dummies(y_train).columns
y_test = raw_test_data['Final_Label']
y_test_onehot = pd.DataFrame(lb.transform(y_test))
y_test_onehot.columns = pd.get_dummies(y_train).columns
#Load model and test on data
loaded_model = tf.keras.models.load_model('./Image_Recog')
y_pred = loaded_model.predict(X_test_final)
y_pred = pd.DataFrame(y_pred)
y_pred.columns = y_test_onehot.columns
ypred_final = y_pred.idxmax(axis='columns')
#Evaluate accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_true=raw_test_data['Final_Label'],y_pred=ypred_final)
#Evaluate precision and recall
from sklearn.metrics import precision_recall_fscore_support as prfs
x = prfs(y_true=y_test,y_pred=ypred_final,
         labels=list(pd.get_dummies(y_train).columns),average=None)
precision = x[0]
precision = pd.DataFrame(precision, index=pd.get_dummies(y_train).columns,
                         columns=['Precision'])
recall = x[1]
recall = pd.DataFrame(recall, index=pd.get_dummies(y_train).columns,
                         columns=['Recall'])
frequency = x[3]
frequency = pd.DataFrame(frequency, index=pd.get_dummies(y_train).columns,
                         columns=['Frequency'])
#Merge into one dataframe
results = pd.concat([precision,recall,frequency],axis=1,join='inner')
results.reset_index(level=None,inplace=True)


