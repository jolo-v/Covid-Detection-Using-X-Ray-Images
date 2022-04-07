# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 21:50:34 2021

@author: jvillanuev29
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

#Define image loading function
def load_data(directory,ids):   
    images = []
    for i in tqdm(range(ids.shape[0])):
        img = keras.preprocessing.image.load_img(directory+ids[i],target_size=(200,200,3))
        img = keras.preprocessing.image.img_to_array(img)
        img = img/255
        images.append(img)
    array = np.array(images)
    return array
def main():
    #Read csv file
    raw_targets = pd.read_csv('Chest_xray_Corona_Metadata.csv')
    raw_targets = raw_targets.iloc[:,1:]
    #Descriptives
    raw_targets[raw_targets['Dataset_type']=='TRAIN'].groupby('Label').size()
    raw_targets[raw_targets['Dataset_type']=='TEST'].groupby('Label').size()
    
    raw_targets[raw_targets['Dataset_type']=='TRAIN'].groupby('Label_2_Virus_category').size()
    raw_targets[raw_targets['Dataset_type']=='TEST'].groupby('Label_2_Virus_category').size()
    
    raw_targets[raw_targets['Dataset_type']=='TRAIN'].groupby('Label_1_Virus_category').size()
    raw_targets[raw_targets['Dataset_type']=='TEST'].groupby('Label_1_Virus_category').size()
    #Change na into blank
    raw_targets.loc[raw_targets['Label_2_Virus_category'].isnull(),'Label_2_Virus_category'] = ''
    raw_targets.loc[raw_targets['Label_1_Virus_category'].isnull(),'Label_1_Virus_category'] = ''
    #Create final label
    raw_targets['Final_Label'] = raw_targets['Label'] + '_' + raw_targets['Label_1_Virus_category'] + '_'+raw_targets['Label_2_Virus_category']
    raw_targets[raw_targets['Dataset_type']=='TRAIN'].groupby('Final_Label').size()
    raw_targets[raw_targets['Dataset_type']=='TEST'].groupby('Final_Label').size()
    #Split covid cases and place some in test set
    covid_indexes = raw_targets.loc[(raw_targets['Final_Label']=='Pnemonia_Virus_COVID-19'),:].index.tolist()
    random.seed(25)
    covid_indexes_test = random.sample(covid_indexes,int((len(covid_indexes))*0.25))
    covid_test = raw_targets.iloc[covid_indexes_test,:]
    #Separate train and test
    train_ids = raw_targets[raw_targets['Dataset_type']=='TRAIN']
    test_ids = raw_targets[raw_targets['Dataset_type']=='TEST']
    train_ids = train_ids.drop(covid_indexes_test,axis=0)
    train_ids.reset_index(level=None,inplace=True,drop=True)
    test_ids.reset_index(level=None,inplace=True,drop=True)
    covid_test.reset_index(level=None,inplace=True,drop=True)
    #Oversample covid images in train set
    covid_train_oversample = train_ids[train_ids['Final_Label']=='Pnemonia_Virus_COVID-19'].sample(n=500,replace=True,random_state=69)
    train_ids_final = pd.concat([train_ids,covid_train_oversample],axis=0)
    train_ids_final = train_ids_final.sample(frac=1,replace=False)
    train_ids_final.reset_index(level=None,inplace=True,drop=True)
    train_ids_final.to_csv('chestxray_corona_train_metadata.csv')
    #Load train and testimages
    X_train = load_data('Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/',train_ids_final['X_ray_image_name'])
    X_test = load_data('Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/',test_ids['X_ray_image_name'])    
    X_test_covid = load_data('Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/',covid_test['X_ray_image_name'])
    X_test_final = np.concatenate((X_test, X_test_covid),axis=0)
    #Look at images
    plt.imshow(X_test_final[562])
    #Concat test ids
    test_ids_final = pd.concat([test_ids,covid_test],axis=0)
    #define y
    from sklearn.preprocessing import LabelBinarizer as LB
    y_train = train_ids_final['Final_Label']
    lb = LB()
    y_train_onehot = pd.DataFrame(lb.fit_transform(y_train))
    y_train_onehot.columns = pd.get_dummies(y_train).columns
    y_test = test_ids_final['Final_Label']
    y_test_onehot = pd.DataFrame(lb.transform(y_test))
    y_test_onehot.columns = pd.get_dummies(y_train).columns
    #Rotate, transform, etc the training set
    datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.2,
            shear_range=0.2,
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip = True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
    datagen.fit(X_train)
    #Build model
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(200,200,3)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=128, activation="swish"))
    model.add(keras.layers.Dense(units=7, activation="softmax"))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    #Train model
    model.fit(datagen.flow(X_train, y_train_onehot, batch_size=64),
             epochs=10)
    #Predict test
    y_pred = model.predict(X_test_final)
    y_pred = pd.DataFrame(y_pred)
    y_pred.columns = y_test_onehot.columns
    ypred_final = y_pred.idxmax(axis='columns')
    #model.evaluate(x=X_test_final,y=y_test_onehot)
    #Evaluate
    from sklearn.metrics import accuracy_score
    accuracy_score(y_true=y_test,y_pred=ypred_final)
    from sklearn.metrics import precision_recall_fscore_support as prfs
    x = prfs(y_true=y_test,y_pred=ypred_final,
             labels=list(train_ids.groupby('Final_Label').size().index),average=None)
    precision = x[0]
    precision = pd.DataFrame(precision, index=train_ids.groupby('Final_Label').size().index,
                             columns=['Precision'])
    recall = x[1]
    recall = pd.DataFrame(recall, index=train_ids.groupby('Final_Label').size().index,
                             columns=['Recall'])
    frequency = x[3]
    frequency = pd.DataFrame(frequency, index=train_ids.groupby('Final_Label').size().index,
                             columns=['Frequency'])
    #Merge into one dataframe
    results = pd.concat([precision,recall,frequency],axis=1,join='inner')
    results.reset_index(level=None,inplace=True)
    #Save model
    tf.saved_model.save(model, export_dir='./Image_Recog')
    #Prepare test set
    test_ids_final.to_csv('chestxray_corona_test_metadata.csv')
    
if __name__ == '__main__':
    main()
    
    