# Import modules
import pandas as pd
import numpy as np
import math as mt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,max_error,explained_variance_score,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import shap
import time
import pickle
from joblib import dump, load
import os
import sys
from contextlib import redirect_stdout
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D,Cropping2D, AveragePooling2D,Dense,Flatten,Reshape,Dropout,TimeDistributed,LSTM,LeakyReLU,RepeatVector
from tensorflow.keras.models import Model,Sequential
import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# # Define the file name for the log
# log_file = 'log_HPT_Reg_xgb_ak0_tilde.log'

# # Function to redirect stdout to the log file
# def redirect_stdout_to_log(log_file):
#     sys.stdout = open(log_file, 'a', buffering=1)

# # Redirect stdout to the log file
# redirect_stdout_to_log(log_file)

############################################################

num_simulations = 152
sequence_length = 101
num_parameters = 6
prediction_window = 20
num_windows = sequence_length - prediction_window

################################## Reading csv ##################################

dataset = pd.read_csv("../../../compiled_data/data_params_HST.csv", engine='python')
parameters_train = dataset[['Time','epsilon','rho_r','mu_r','La_l','Bo_l']]

scaler = preprocessing.MinMaxScaler()
scaler.fit(parameters_train)

#parameters_train_scaled = scaler.transform(parameters_train)

parameters_train_scaled = parameters_train.values

#parameters_train_scaled = parameters_train_scaled.values
train_parameter = parameters_train_scaled.reshape(num_simulations,sequence_length,num_parameters)

ak0_tilde_train = dataset[['ak0_tilde']]
ak0_tilde_train = ak0_tilde_train.values
ak0_in = ak0_tilde_train.reshape(num_simulations,sequence_length,1)

############################################################

input_seq = np.zeros((1,prediction_window,1))
for i in range(num_simulations):
  current_traj = ak0_in[i,:,:]

  for start in range(num_windows):
    input_seq = np.concatenate((input_seq,current_traj[start:(start+prediction_window),:].reshape(1,prediction_window,1)),axis=0)

input_seq = input_seq[1:,:,:]
print(input_seq.shape)

############################################################

train_parameter_shifted = np.zeros((num_simulations * (sequence_length - prediction_window), num_parameters))

index = 0
for i in range(num_simulations):
    for j in range(num_windows):
        shifted_parameters = train_parameter[i, j + 1 : j + 1 + prediction_window, :]
        train_parameter_shifted[index] = shifted_parameters[-1]
        index += 1

print(train_parameter_shifted.shape)
################################################################

input_dim = 1
hidden_size=100

use_dropout=True

model = Sequential()

model.add(LSTM(hidden_size,input_shape=(prediction_window,input_dim),activation='relu'))

model.add(Dense(64))
#model.add(Dense(100))
model.add(LeakyReLU(alpha=0.2))
model.add(Dense(num_parameters))
#model.add(LeakyReLU(alpha=0.2))

#optimizer = optimizers.Adam()
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
history = model.fit(input_seq , train_parameter_shifted, validation_split=0.05, epochs=10,batch_size=64,verbose=2)

predicted_parameters = model.predict(input_seq[0,:,:].reshape(1,prediction_window,1))
print(predicted_parameters)
