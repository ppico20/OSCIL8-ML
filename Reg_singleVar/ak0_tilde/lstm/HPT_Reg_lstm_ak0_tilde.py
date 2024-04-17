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

# # Define the file name for the log
# log_file = 'log_HPT_Reg_xgb_ak0_tilde.log'

# # Function to redirect stdout to the log file
# def redirect_stdout_to_log(log_file):
#     sys.stdout = open(log_file, 'a', buffering=1)

# # Redirect stdout to the log file
# redirect_stdout_to_log(log_file)

################################## Reading csv ##################################

dataset = pd.read_csv("../../../compiled_data/data_params_HST.csv", engine='python')


# data = dataset[['Time_tilde','epsilon','rho_r','mu_r','La_l','Bo_l']]
# columns = ['Time_tilde','epsilon','rho_r','mu_r','La_l','Bo_l']

# # Selecting labels
# label = dataset[['ak0_tilde']]

parameters_train = dataset[['Time_tilde','epsilon','rho_r','mu_r','La_l','Bo_l']]
parameters_train = parameters_train.values
train_parameter = parameters_train.reshape(152,101,6)

ak0_tilde_train = dataset[['ak0_tilde']]
ak0_tilde_train = ak0_tilde_train.values
ak0_in = ak0_tilde_train.reshape(152,101,1)

#we train a four to four LSTM

input_seq = np.zeros((1,10,1))
for i in range(152):
  current_traj = ak0_in[i,:,:]

  for start in range(91):
    input_seq = np.concatenate((input_seq,current_traj[start:(start+10),:].reshape(1,10,1)),axis=0)


num_simulations = 152
sequence_length = 101
num_parameters = 6
prediction_window = 10  # Predict parameters for the next 4 time steps

# Initialize a new array for shifted parameters
train_parameter_shifted = np.zeros((num_simulations * (sequence_length - prediction_window), num_parameters))

# Fill train_parameter_shifted with shifted parameter sequences
index = 0
for i in range(num_simulations):
    for j in range(sequence_length - prediction_window):  # Iterate up to (20 - 4) = 16
        # Extract parameters for the next 4 time steps (shifted forward)
        shifted_parameters = train_parameter[i, j + 1 : j + 1 + prediction_window, :]
        # Flatten and assign to train_parameter_shifted
        train_parameter_shifted[index] = shifted_parameters[-1]  # Use parameters from the last time step
        index += 1

# Verify the shape of train_parameter_shifted
print(train_parameter_shifted.shape)

################################################################

####################################################################################################
input_seq = input_seq[1:,:,:]
# output_seq = output_seq[1:,:,:]


print(input_seq.shape)
# #training sample: 12000, number of steps in each sample: 4, size of the latent space: 30

print(train_parameter_shifted.shape)


input_dim = 1
n_memory = 10 #input sequence length
output_dim = 6 #number of parameters

hidden_size=100

use_dropout=True

model = Sequential()

model.add(LSTM(hidden_size,input_shape=(n_memory,input_dim),activation='relu'))

model.add(Dense(64))
#model.add(Dense(100))
model.add(LeakyReLU(alpha=0.2))
model.add(Dense(output_dim))
#model.add(LeakyReLU(alpha=0.2))

#optimizer = optimizers.Adam()
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
history = model.fit(input_seq , train_parameter_shifted, validation_split=0.05, epochs=10,batch_size=64,verbose=2)

predicted_parameters = model.predict(input_seq[2,:,:].reshape(1,10,1))
print(predicted_parameters)
