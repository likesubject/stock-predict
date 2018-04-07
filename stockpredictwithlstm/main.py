import tensorflow as tf

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import LSTM, Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
import tushare as ts
import pandas  as pd
from sklearn import preprocessing
import numpy as np
from stockdata import *

Target = ['300024', '002230', '600460', '600570', '600000', '002415', '601318', '600028']
Max_Tx, Max_Feature = creat_stock_csv(Target)
Norm_Data, Minmax_Data, Sample_Num = creat_stock_data(Target)

Win_Tx = Max_Tx -1
Input_Feature = Max_Feature
Pred_Day = 1

def train_lab_pre_data(norm_data, win_tx, pred_day=1):
     train_data = []
     lab_data = []
     pre_data = []
     for data in norm_data:
         trdata = data[:-pred_day,:]
         lab_data.append(data[-pred_day,2])         #choose close price as feature
         prdata = data[1:,:]
         xn, fn = np.shape(trdata)
         if xn < win_tx - pred_day:
             mask = np.ones((win_tx - pred_day - xn, fn)).astype('float32') * -1.0
             trdata = np.row_stack((trdata, mask))
         train_data.append(trdata)
         xn, fn = np.shape(prdata)
         if xn < win_tx - pred_day:
             mask = np.ones((win_tx - pred_day - xn, fn)).astype('float32') * -1.0
             prdata = np.row_stack((prdata, mask))
         pre_data.append(prdata)
     train_data = np.array(train_data)
     pre_data = np.array(pre_data)
     lab_data = np.array(lab_data).reshape((8,1))
     print(np.shape(train_data))
     print(np.shape(lab_data))
     print(np.shape(pre_data))
     tx=np.shape(train_data)[1]
     feature = np.shape(train_data)[2]

     return train_data, tx, feature, lab_data, pre_data

train_data, Win_Tx, Input_Feature, lab_data, pre_data  = train_lab_pre_data(Norm_Data, Win_Tx, Pred_Day)



input_shape = (Win_Tx, Input_Feature)



def creat_lstm_model(input_shape, unit_num=64, pred_day=1, drop_rate=0.5):

#     X = Input(shape=input_shape,dtype='float32')
#     X = Masking(mask_value=-1)(X)
# #    Xin = BatchNormalization()(Xin)
#     X = LSTM(unit_num,return_sequences=True)(X)
#     X = Dropout(drop_rate)(X)
#     X = BatchNormalization()(X)
#     X = LSTM(unit_num,return_sequences=True)(X)
#     X = Dropout(drop_rate)(X)
#     X = BatchNormalization()(X)
#     X = LSTM(unit_num)(X)
#     X = Dropout(drop_rate)(X)
#     X = BatchNormalization()(X)
#     PRE = Dense(pred_day, activation='relu')(X)

    model = Sequential()
    model.add(Masking(mask_value=-1, input_shape=input_shape))
    model.add(LSTM(unit_num, return_sequences=True))
    model.add(Dropout(drop_rate))
    model.add(BatchNormalization())
    model.add(LSTM(unit_num, return_sequences=True))
    model.add(Dropout(drop_rate))
    model.add(BatchNormalization())
    model.add(LSTM(unit_num))
    model.add(Dropout(drop_rate))
    model.add(BatchNormalization())
    model.add(Dense(pred_day,activation='relu'))
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
    return model

lstm_model = creat_lstm_model(input_shape)
#lstm_model.summary()

lstm_model.fit(train_data, lab_data, batch_size = 1, epochs=1)
lstm_model.save('lstm_model.h5')
predict_out=lstm_model.predict(pre_data, batch_size=1)
predict_val=np.zeros((Sample_Num, Pred_Day))
for i in range(Sample_Num):
    predict_val[i] = predict_out[i]*Minmax_Data[i].data_range_[2]+Minmax_Data[i].data_min_[2]
