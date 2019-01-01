import IPython.display as ipd
import os
import pandas as pd
import librosa
import glob
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder

class SpeechProcesser:

    train_X = []
    train_y = []
    valid_X = []
    valid_y = []


    def __init__(self, train_name, val_name):
        df1=pd.read_csv(train_name, sep=';')
        df2=pd.read_csv(val_name, sep=';')
        
        self.train_wavnames=df1['filenames'].values.tolist()
        self.train_labels=df1['labels'].values.tolist()

        self.valid_wavnames=df2['filenames'].values.tolist()
        self.valid_labels=df2['labels'].values.tolist()
        

        for i in range(0, len(self.train_wavnames)):
            self.train_wavnames[i] = str(self.train_wavnames[i]).strip() + '.wav'.strip()
        for i in range(0, len(self.valid_wavnames)):
            self.valid_wavnames[i] = str(self.valid_wavnames[i]).strip() + '.wav'.strip()

        self.train_y = self.train_labels
        self.valid_y = self.valid_labels
        print("Tanito adat: " + str(len(self.train_wavnames)))
        print("Tanito minta: " + str(len(self.train_y)))
        print("Validacios adat: " + str(len(self.valid_wavnames)))
        print("Validacios minta: " + str(len(self.valid_y)))

    def parser(self, fname, labelname):
        # function to load files and extract features
        file_name = fname  #megadjuk a parameterben kapott fajlnevet

        # handle exception to check if there isn't a file which is corrupted
        # here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        # we extract mfcc feature from data
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=128).T,axis=0) 
    
    
        feature = mfccs
        label = labelname
    
        return [feature, label]

    def create_train_datas(self):

        for i in range(0,len(self.train_wavnames)):
            parsedData = self.parser("train/" + str(self.train_wavnames[i]), self.train_y[i])
            self.train_X.append(parsedData[0])
            
        for i in range(0, len(self.valid_wavnames)):
            parsedData = self.parser("dev/" + str(self.valid_wavnames[i]), self.valid_y[i])
            self.valid_X.append(parsedData[0])
            


    def evaluate(self):
        lb = LabelEncoder()
        t_X = np.array(self.train_X)
        t_y = np.array(self.train_y)
        t_y = np_utils.to_categorical(lb.fit_transform(t_y))
        
        v_X = np.array(self.valid_X)
        v_y = np.array(self.valid_y)
        v_y = np_utils.to_categorical(lb.fit_transform(v_y))

        print("Tanito adat: " + str(len(t_X)))
        print("Tanito minta: " + str(len(t_y)))
        print("Validacios adat: " + str(len(v_X)))
        print("Validacios minta: " + str(len(v_y)))

        num_labels = t_y.shape[1]
        print(num_labels)

        model = Sequential()
        model.add(Dense(1000, input_shape=(128,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(1000))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(num_labels))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
        model.fit(t_X, t_y, batch_size=686, epochs=25, validation_data=(v_X, v_y))

obj = SpeechProcesser('labels_train.csv', 'labels_dev.csv')
obj.create_train_datas()
obj.evaluate()