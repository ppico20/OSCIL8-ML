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
num_simulations_train = 120
num_simulations_test = 152 - num_simulations_train
sequence_length = 101
output_dim = 6
input_dim = 4
prediction_window = 20
num_windows = sequence_length - prediction_window

################################## Reading csv and Train/test split ##################################

dataset = pd.read_csv("../../../compiled_data/data_params_HST.csv", engine='python')
unique_cases = dataset['Run_ID'].unique()
np.random.seed(42)
np.random.shuffle(unique_cases)

train_cases = unique_cases[:num_simulations_train]
test_cases = unique_cases[num_simulations_train:]
train_set = dataset[dataset['Run_ID'].isin(train_cases)]
test_set = dataset[dataset['Run_ID'].isin(test_cases)]

X_train = train_set[['ak0_tilde','ak1_tilde','ak2_tilde','ak3_tilde']]
X_test = test_set[['ak0_tilde','ak1_tilde','ak2_tilde','ak3_tilde']]
Y_train = train_set[['Time_tilde','epsilon','rho_r','mu_r','La_l','Bo_l']]
Y_test = test_set[['Time_tilde','epsilon','rho_r','mu_r','La_l','Bo_l']]

################################## Selecting input/output variables ##################################

scaler = preprocessing.MinMaxScaler()
scaler.fit(Y_train)
Y_train_scaled = scaler.transform(Y_train)
Y_test_scaled = scaler.transform(Y_test)

Y_train_scaled = Y_train_scaled.reshape(num_simulations_train,sequence_length,output_dim)
Y_test_scaled = Y_test_scaled.reshape(num_simulations_test,sequence_length,output_dim)

X_train = X_train.values.reshape(num_simulations_train,sequence_length,input_dim)
X_test = X_test.values.reshape(num_simulations_test,sequence_length,input_dim)

################################## Creating input sequences ##################################

def create_input_seq(input_data,num_simulations):
  input_seq = np.zeros((1,prediction_window,input_dim))
  for i in range(num_simulations):
    current_traj = input_data[i,:,:]

    for start in range(num_windows):
      input_seq = np.concatenate((input_seq,current_traj[start:(start+prediction_window),:].reshape(1,prediction_window,input_dim)),axis=0)

  input_seq = input_seq[1:,:,:]
  return input_seq
   
input_seq_train = create_input_seq(X_train,num_simulations_train)
input_seq_test = create_input_seq(X_test,num_simulations_test)
print(input_seq_train.shape)

################################## Creating output sequences ##################################

def create_output_seq(output_data,num_simulations):
  output_seq = np.zeros((num_simulations * (sequence_length - prediction_window), output_dim))
  index = 0
  for i in range(num_simulations):
    for j in range(num_windows):
        shifted_values = output_data[i, j : j + prediction_window, :]
        output_seq[index] = shifted_values[0]
        index += 1
  return output_seq

output_seq_train = create_output_seq(Y_train_scaled,num_simulations_train)
output_seq_test = create_output_seq(Y_test_scaled,num_simulations_test)
print(output_seq_train.shape)

################################## LSTM ##################################

hidden_size = 100
use_dropout = True

model = Sequential()

model.add(LSTM(hidden_size,input_shape=(prediction_window,input_dim),activation='relu'))

model.add(Dense(64))
model.add(Dense(100))
model.add(LeakyReLU(alpha=0.2))
model.add(Dense(output_dim))
model.add(LeakyReLU(alpha=0.2))

#optimizer = optimizers.Adam()
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
history = model.fit(input_seq_train , output_seq_train, validation_split=0.05, epochs = 20,batch_size=64,verbose=2)

# #################################### Evaluate model ##############################################################

# predicted_parameters = model.predict(input_seq_train[0:10,:,:].reshape(10,prediction_window,input_dim))
# print(predicted_parameters)

y_pred_train = model.predict(input_seq_train)
y_pred_test = model.predict(input_seq_test)

print('############ Performance metrics in training set ################')
print("Coefficient of determination, r2 = %.5f" % r2_score(output_seq_train, y_pred_train))
print("Mean Absolute Error, MAE = %.5f" % mean_absolute_error(output_seq_train, y_pred_train))
print("Mean squared error, MSE = %.5f" % mean_squared_error(output_seq_train, y_pred_train))
print("Explained Variance Score = %.5f" % explained_variance_score(output_seq_train, y_pred_train))

print('############ Performance metrics in testing set ################')
print("Coefficient of determination, r2 = %.5f" % r2_score(output_seq_test, y_pred_test))
print("Mean Absolute Error, MAE = %.5f" % mean_absolute_error(output_seq_test, y_pred_test))
print("Mean squared error, MSE = %.5f" % mean_squared_error(output_seq_test, y_pred_test))
print("Explained Variance Score = %.5f" % explained_variance_score(output_seq_test, y_pred_test))