# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:24:22 2017

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from keras.models import Sequential
from keras.layers import Dense,UpSampling1D
from keras.layers import LSTM
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten,Activation,BatchNormalization,Input
from keras.layers.pooling import MaxPooling1D
from keras.layers import Conv2D, MaxPooling2D,Conv1D
from sklearn.model_selection import ShuffleSplit
from keras.optimizers import Adam,RMSprop,Adadelta
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
from sklearn.metrics import mean_squared_error,mean_absolute_error
import theano.tensor as T
from sklearn.utils import check_array
from keras import regularizers
from sklearn.linear_model import LinearRegression,Lasso
check = 1
pd.set_option('display.max_columns', 322)
#dataset_avg_time = pd.read_csv("DATA_WC3_COMPLETE.csv",  encoding = "ISO-8859-1")
energy_aemo_jan_nsw = pd.read_csv(r'C:\kaggle\thesis\sheraz\evaluation data set\filtered\latest\1\jan\PRICE_AND_DEMAND_201301_NSW1.csv').drop(['SETTLEMENTDATE','REGION','RRP','PERIODTYPE'], axis=1)
energy_aemo_april_nsw = pd.read_csv(r'C:\kaggle\thesis\sheraz\evaluation data set\filtered\latest\1\april\PRICE_AND_DEMAND_201304_NSW1.csv').drop(['SETTLEMENTDATE','REGION','RRP','PERIODTYPE'], axis=1)
energy_aemo_jul_nsw = pd.read_csv(r'C:\kaggle\thesis\sheraz\evaluation data set\filtered\latest\1\julz\PRICE_AND_DEMAND_201307_NSW1.csv').drop(['SETTLEMENTDATE','REGION','RRP','PERIODTYPE'], axis=1)
energy_aemo_oct_nsw = pd.read_csv(r'C:\kaggle\thesis\sheraz\evaluation data set\filtered\latest\1\oct\PRICE_AND_DEMAND_201310_NSW1.csv').drop(['SETTLEMENTDATE','REGION','RRP','PERIODTYPE'], axis=1)

energy_aemo_jan_tas = pd.read_csv(r'C:\kaggle\thesis\sheraz\evaluation data set\filtered\latest\1\jan\PRICE_AND_DEMAND_201301_TAS1.csv').drop(['SETTLEMENTDATE','REGION','RRP','PERIODTYPE'], axis=1)
energy_aemo_april_tas = pd.read_csv(r'C:\kaggle\thesis\sheraz\evaluation data set\filtered\latest\1\april\PRICE_AND_DEMAND_201304_TAS1.csv').drop(['SETTLEMENTDATE','REGION','RRP','PERIODTYPE'], axis=1)
energy_aemo_jul_tas = pd.read_csv(r'C:\kaggle\thesis\sheraz\evaluation data set\filtered\latest\1\julz\PRICE_AND_DEMAND_201307_TAS1.csv').drop(['SETTLEMENTDATE','REGION','RRP','PERIODTYPE'], axis=1)
energy_aemo_oct_tas = pd.read_csv(r'C:\kaggle\thesis\sheraz\evaluation data set\filtered\latest\1\oct\PRICE_AND_DEMAND_201310_TAS1.csv').drop(['SETTLEMENTDATE','REGION','RRP','PERIODTYPE'], axis=1)

energy_aemo_jan_qld = pd.read_csv(r'C:\kaggle\thesis\sheraz\evaluation data set\filtered\latest\1\jan\PRICE_AND_DEMAND_201301_QLD1.csv').drop(['SETTLEMENTDATE','REGION','RRP','PERIODTYPE'], axis=1)
energy_aemo_april_qld = pd.read_csv(r'C:\kaggle\thesis\sheraz\evaluation data set\filtered\latest\1\april\PRICE_AND_DEMAND_201304_QLD1.csv').drop(['SETTLEMENTDATE','REGION','RRP','PERIODTYPE'], axis=1)
energy_aemo_jul_qld = pd.read_csv(r'C:\kaggle\thesis\sheraz\evaluation data set\filtered\latest\1\julz\PRICE_AND_DEMAND_201307_QLD1.csv').drop(['SETTLEMENTDATE','REGION','RRP','PERIODTYPE'], axis=1)
energy_aemo_oct_qld = pd.read_csv(r'C:\kaggle\thesis\sheraz\evaluation data set\filtered\latest\1\oct\PRICE_AND_DEMAND_201310_QLD1.csv').drop(['SETTLEMENTDATE','REGION','RRP','PERIODTYPE'], axis=1)

energy_aemo_jan_vic = pd.read_csv(r'C:\kaggle\thesis\sheraz\evaluation data set\filtered\latest\1\jan\PRICE_AND_DEMAND_201301_VIC1.csv').drop(['SETTLEMENTDATE','REGION','RRP','PERIODTYPE'], axis=1)
energy_aemo_april_vic = pd.read_csv(r'C:\kaggle\thesis\sheraz\evaluation data set\filtered\latest\1\april\PRICE_AND_DEMAND_201304_VIC1.csv').drop(['SETTLEMENTDATE','REGION','RRP','PERIODTYPE'], axis=1)
energy_aemo_jul_vic = pd.read_csv(r'C:\kaggle\thesis\sheraz\evaluation data set\filtered\latest\1\julz\PRICE_AND_DEMAND_201307_VIC1.csv').drop(['SETTLEMENTDATE','REGION','RRP','PERIODTYPE'], axis=1)
energy_aemo_oct_vic = pd.read_csv(r'C:\kaggle\thesis\sheraz\evaluation data set\filtered\latest\1\oct\PRICE_AND_DEMAND_201310_VIC1.csv').drop(['SETTLEMENTDATE','REGION','RRP','PERIODTYPE'], axis=1)

energy_aemo_jan_sa = pd.read_csv(r'C:\kaggle\thesis\sheraz\evaluation data set\filtered\latest\1\jan\PRICE_AND_DEMAND_201301_SA1.csv').drop(['SETTLEMENTDATE','REGION','RRP','PERIODTYPE'], axis=1)
energy_aemo_april_sa = pd.read_csv(r'C:\kaggle\thesis\sheraz\evaluation data set\filtered\latest\1\april\PRICE_AND_DEMAND_201304_SA1.csv').drop(['SETTLEMENTDATE','REGION','RRP','PERIODTYPE'], axis=1)
energy_aemo_jul_sa = pd.read_csv(r'C:\kaggle\thesis\sheraz\evaluation data set\filtered\latest\1\julz\PRICE_AND_DEMAND_201307_SA1.csv').drop(['SETTLEMENTDATE','REGION','RRP','PERIODTYPE'], axis=1)
energy_aemo_oct_sa = pd.read_csv(r'C:\kaggle\thesis\sheraz\evaluation data set\filtered\latest\1\oct\PRICE_AND_DEMAND_201310_SA1.csv').drop(['SETTLEMENTDATE','REGION','RRP','PERIODTYPE'], axis=1)

alldataset = [energy_aemo_jan_nsw,energy_aemo_april_nsw,energy_aemo_jul_nsw,energy_aemo_oct_nsw,energy_aemo_jan_tas,energy_aemo_april_tas,energy_aemo_jul_tas,energy_aemo_oct_tas,energy_aemo_jan_qld,energy_aemo_april_qld,energy_aemo_jul_qld,energy_aemo_oct_qld,energy_aemo_jan_vic,energy_aemo_april_vic,energy_aemo_jul_vic,energy_aemo_oct_vic,energy_aemo_jan_sa,energy_aemo_april_sa,energy_aemo_jul_sa,energy_aemo_oct_sa]
alldataset_names = ['energy_aemo_jan_nsw','energy_aemo_april_nsw','energy_aemo_jul_nsw','energy_aemo_oct_nsw','energy_aemo_jan_tas','energy_aemo_april_tas','energy_aemo_jul_tas','energy_aemo_oct_tas','energy_aemo_jan_qld','energy_aemo_april_qld','energy_aemo_jul_qld','energy_aemo_oct_qld','energy_aemo_jan_vic','energy_aemo_april_vic','energy_aemo_jul_vic','energy_aemo_oct_vic','energy_aemo_jan_sa','energy_aemo_april_sa','energy_aemo_jul_sa','energy_aemo_oct_sa']


#alldataset = [energy_aemo_jan_nsw,energy_aemo_april_nsw,energy_aemo_jul_nsw,energy_aemo_oct_nsw,energy_aemo_jan_tas,energy_aemo_april_tas,energy_aemo_jul_tas,energy_aemo_oct_tas,energy_aemo_jan_qld,energy_aemo_april_qld,energy_aemo_jul_qld,energy_aemo_oct_qld,energy_aemo_jan_vic,energy_aemo_april_vic,energy_aemo_jul_vic,energy_aemo_oct_vic,energy_aemo_jan_sa,energy_aemo_april_sa,energy_aemo_jul_sa,energy_aemo_oct_sa,energy_aemo_jan_nsw,energy_aemo_april_nsw,energy_aemo_jul_nsw,energy_aemo_oct_nsw,energy_aemo_jan_tas,energy_aemo_april_tas,energy_aemo_jul_tas,energy_aemo_oct_tas,energy_aemo_jan_qld,energy_aemo_april_qld,energy_aemo_jul_qld,energy_aemo_oct_qld,energy_aemo_jan_vic,energy_aemo_april_vic,energy_aemo_jul_vic,energy_aemo_oct_vic,energy_aemo_jan_sa,energy_aemo_april_sa,energy_aemo_jul_sa,energy_aemo_oct_sa]
#alldataset_names = ['energy_aemo_jan_nsw','energy_aemo_april_nsw','energy_aemo_jul_nsw','energy_aemo_oct_nsw','energy_aemo_jan_tas','energy_aemo_april_tas','energy_aemo_jul_tas','energy_aemo_oct_tas','energy_aemo_jan_qld','energy_aemo_april_qld','energy_aemo_jul_qld','energy_aemo_oct_qld','energy_aemo_jan_vic','energy_aemo_april_vic','energy_aemo_jul_vic','energy_aemo_oct_vic','energy_aemo_jan_sa','energy_aemo_april_sa','energy_aemo_jul_sa','energy_aemo_oct_sa','energy_aemo_jan_nsw','energy_aemo_april_nsw','energy_aemo_jul_nsw','energy_aemo_oct_nsw','energy_aemo_jan_tas','energy_aemo_april_tas','energy_aemo_jul_tas','energy_aemo_oct_tas','energy_aemo_jan_qld','energy_aemo_april_qld','energy_aemo_jul_qld','energy_aemo_oct_qld','energy_aemo_jan_vic','energy_aemo_april_vic','energy_aemo_jul_vic','energy_aemo_oct_vic','energy_aemo_jan_sa','energy_aemo_april_sa','energy_aemo_jul_sa','energy_aemo_oct_sa']



alldataset = [energy_aemo_april_nsw]
alldataset_names = ['energy_aemo_april_nsw']

check_weight_saved = 0
count = 0
dnn = 1
def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def mean_absolute_percentage_error(y_true, y_pred):
    #y_true, y_pred = check_array(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))*100


def scheduler(epoch):
    initial_lrate = float(0.001)
    if epoch > 50:
        return float(0.0001)
    else:
        return initial_lrate


for alldata in alldataset:
    dataframe = alldata
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    #scaler = MinMaxScaler(feature_range=(0, 1))
    if(dnn ==1):
        scaler = StandardScaler()
        dataset = scaler.fit_transform(dataset)
    
    look_back = 10
    x_dataset, y_dataset = create_dataset(dataset, look_back)
    

    
    train_size = int(x_dataset.shape[0] * 0.75)
    test_size = x_dataset.shape[0] - train_size
    x_train, x_test = x_dataset[0:train_size,:], x_dataset[train_size:,:]
    y_train, y_test = y_dataset[0:train_size], y_dataset[train_size:]
    if(dnn == 0):
        y_true = y_test
        
        
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        y_pred_lr = lr.predict(x_test)
        
        mape = mean_absolute_percentage_error(y_true, y_pred_lr)
        print(mape)
        RMSE = mean_squared_error(y_true, y_pred_lr)**0.5
        print(RMSE)
        
        rf = RandomForestRegressor(n_estimators  = 100, n_jobs = 1)
        rf.fit(x_train, y_train)
        y_pred_rf = rf.predict(x_test)
        
        mape = mean_absolute_percentage_error(y_true, y_pred_rf)
        print(mape)
        RMSE = mean_squared_error(y_true, y_pred_rf)**0.5
        print(RMSE)
        
        mlp = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100,30),  random_state=1,max_iter =100)
        mlp.fit(x_train, y_train)
        y_pred_mlp = mlp.predict(x_test)
        
        mape = mean_absolute_percentage_error(y_true, y_pred_mlp)
        print(mape)
        RMSE = mean_squared_error(y_true, y_pred_mlp)**0.5
        print(RMSE)
        

        
        y_pred_nm = np.hstack((y_train[-1],y_test[0:-1]))
        
        
        pd.DataFrame(np.vstack([y_pred_nm, y_pred_lr,y_pred_rf,y_pred_mlp])).T.to_csv(path_or_buf= 'all_for_graph.csv', index = False)

    
    
    '''   validation split'''
    
    if(dnn == 1):
    
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.33, random_state=42)
        
        
        
        batch_size = 1
        # reshape input to be [samples, time steps, features]
        trainX_1 = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
        valX_1 = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
        testX_1 = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        
        
        clf_1 = Sequential()
        clf_1.add(Conv1D(32, kernel_size=3,strides=1,activation='relu',input_shape=(10, 1),kernel_initializer='glorot_normal',name='cnn1',padding='same',trainable=True))
        clf_1.add(Conv1D(64, kernel_size=5,strides=1,activation='relu',name='cnn2',padding='same',trainable=True))
        clf_1.add(Flatten())
        clf_1.add(Dense(30,kernel_initializer='glorot_normal',name='dense_30',trainable=True))
        clf_1.add(Activation('relu')) 
        clf_1.add(Dense(1,kernel_initializer='glorot_normal',name='dense_1a'))
        clf_1.add(Activation('linear'))
        
        rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        adadelta = Adadelta(lr=0.3, rho=0.95, epsilon=1e-08, decay=0.0)
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        clf_1.compile(loss='mean_absolute_error',optimizer=rmsprop)#
        filepath_1_cnn = "/tmp/weights.best_1_cnn.hdf5"
        checkpoint_1_cnn = ModelCheckpoint(filepath_1_cnn, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        change_lr = LearningRateScheduler(scheduler)
        #callbacks_list_1_cnn = [checkpoint_1_cnn,change_lr]
        callbacks_list_1_cnn = [checkpoint_1_cnn]
        if (check_weight_saved==1):
               clf_1.load_weights("/tmp/weights.best_1_cnn.hdf5")
        clf_1.fit(trainX_1, y_train,validation_data=(valX_1,y_val), epochs=50, batch_size=batch_size, verbose=2,callbacks=callbacks_list_1_cnn) # volume
        #clf_1.fit(trainX_1, y_train, epochs=150,verbose=2, batch_size=batch_size) # volume
        clf_1.load_weights("/tmp/weights.best_1_cnn.hdf5")
        #clf_1.save_weights("/tmp/weights.best_1_cnn.hdf5")
        y_pred = clf_1.predict(testX_1,batch_size = batch_size)
        
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_true = scaler.inverse_transform(y_test.reshape(-1, 1))
        pd.DataFrame(y_true).to_csv(path_or_buf= 'true.csv', index = False)
        pd.DataFrame(y_pred).to_csv(path_or_buf= 'pred.csv', index = False)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(mape)
    RMSE = mean_squared_error(y_true, y_pred)**0.5
    print(RMSE)
    #check_weight_saved = 1
    
    
    if(check == 1):
        data_log_array = np.array([alldataset_names[count],mape,RMSE])
        data_log = pd.DataFrame(data_log_array).T
        check = 0
    else: 
        data_log_array = np.array([alldataset_names[count],mape,RMSE])
        temp = pd.DataFrame(data_log_array).T  
        data_log = pd.concat([data_log,temp], axis=0)
        #print(data_log)
    count = count +1

mape_mean = pd.to_numeric(data_log.ix[:, 1]).mean()
rmse_mean = pd.to_numeric(data_log.ix[:, 2]).mean()

data_log_array = np.array(['CNN',mape_mean,rmse_mean])
temp = pd.DataFrame(data_log_array).T  
data_log = pd.concat([data_log,temp], axis=0)



data_log.to_csv(path_or_buf= 'AEMO.csv', index = False)

		
		
