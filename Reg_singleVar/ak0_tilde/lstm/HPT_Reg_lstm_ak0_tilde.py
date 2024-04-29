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
from tensorflow_addons.metrics import RSquare
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import itertools
import csv
from keras.models import load_model

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

################################## Creating LSTM model ##################################

def create_model(hidden_sizes = 100,learning_rates=0.0001,optimizers=keras.optimizers.Adam):
  """
  This function creates the general structure of the network with default hyperparameter values
  """
  model = Sequential()
  model.add(LSTM(hidden_sizes,input_shape=(prediction_window,input_dim),activation='relu'))
  model.add(Dense(64))
  model.add(Dense(100))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dense(output_dim))
  model.add(LeakyReLU(alpha=0.2))
  model.summary()

  model.compile(loss='mean_squared_error', optimizer=optimizers(learning_rate=learning_rates), metrics=[RSquare()])
  return model

################################## Function to train our LSTM for each set of hyperparameters ##################################

def train_model_HPT(hidden_size, learning_rate, optimizer, X_train, y_train, X_val, y_val):
    """
    This function is tasked with training the LSTM on each parameter combination for hyperparameter tunning (on each fold).
    It uses an eary stop on the loss function (mse) of the validation set.
    Returns the trained model, the performance metric on the validation data, and the training history.
    """

    model = create_model(hidden_sizes = hidden_size, learning_rates=learning_rate, optimizers=optimizer)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30, restore_best_weights=True)
    hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, callbacks=[es], batch_size=64, verbose=1)
    val_evaluate = model.evaluate(X_val, y_val, verbose=0)[1]
    return model, hist, val_evaluate

################################## Hyperparameter tuning function with k-fold ##################################
def tune_hyperparameters(hidden_sizes, learning_rates, optimizers_map, X_train, y_train, kfold):
    """
    This function performs hyperparameter tunning. It goes over each list of hyperparameters (activation functions, optimisers,
    etc) and creates a k-fold cross validation for each parameter combination. It returns the model that produced the best average
    r2 (over the k folds), its history, and the best parameters)
    """

    fieldnames = ['Index', 'Hidden Size', 'Learning Rate', 'Optimizer', 'Mean Val Evaluate']
    csvfile = open(output_csv_file, 'w', newline='')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    best_val_evaluate = float('-inf')
    best_params = {}
    tuning_results = []
    param_combinations = itertools.product(hidden_sizes, learning_rates, optimizers_map.keys())
    
    for idx, (hidden_size, learning_rate, optimizer_key) in enumerate(param_combinations):
        print('############################# Testing a new configuration #################################')
        print(f'Parameter Configuration Index: {idx}')
        print(f'Current settings: hidden_size={hidden_size}, learning rate={learning_rate}, optimizer={optimizer_key} \n')
        val_evaluate_list = []
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
            print(f'Currently on fold {fold_idx}')
            try:
                model, hist, val_evaluate = train_model_HPT(hidden_size, learning_rate, optimizers_map[optimizer_key], X_train[train_idx,:,:], y_train[train_idx,:], X_train[val_idx,:,:], y_train[val_idx,:])
                val_evaluate_list.append(val_evaluate)
            except Exception as e:
                print(f"Error occurred for params: hidden_size={hidden_size}, learning_rate={learning_rate}, optimizer={optimizer_key}")
                print(str(e))
                continue
        mean_val_evaluate = np.mean(val_evaluate_list)
        print(f'Average metric value in this configuration (validation data): {mean_val_evaluate}')
        tuning_results.append((hidden_size, learning_rate, optimizer_key, mean_val_evaluate))
        row_data = {'Index': idx, 'Hidden Size': hidden_size, 'Learning Rate': learning_rate, 'Optimizer': optimizer_key, 'Mean Val Evaluate': mean_val_evaluate}
        writer.writerow(row_data)
        csvfile.flush()
        print('############################################################################################# \n')
        
        if mean_val_evaluate > best_val_evaluate:
            best_val_evaluate = mean_val_evaluate
            best_params = {'hidden_size': hidden_size,'learning_rate': learning_rate, 'optimizer': optimizer_key}
            print(f'Best validation metric value: {best_val_evaluate} | Params: {best_params} \n')
            # Save best model and its history
            model.save(best_model_kfold_path)
            history = hist
    
    csvfile.close()
    return model, history, best_params, best_val_evaluate, tuning_results

################################## Declaration of hyperparameters ranges to explore ##################################

# define hyperparameters
hidden_sizes = [100]
learning_rates = [0.0001]
optimizers_map = {'adam': keras.optimizers.Adam}

# Kfold definition
kfold = KFold(n_splits=2, shuffle=True)

################################## Performing hyperparameter tunning ##################################

start_time = time.time()
output_csv_file = 'tuning_lstm_ak0_tilde.csv'
best_model_kfold_path = 'best_model_train_kfold'
model, history, best_params, best_val_evaluate, tuning_results = tune_hyperparameters(hidden_sizes, learning_rates, optimizers_map, input_seq_train, output_seq_train, kfold)
# save_tuning_to_csv(output_csv_file, tuning_results)

end_time = time.time()  
elapsed_time = end_time - start_time

print(f"Hyperparameter tunning completed in {elapsed_time:.2f} seconds.")

################################## Loading models ##################################

print('############################# Loading pre-trained model #################################')

model = load_model(best_model_kfold_path)

################################## Training final model with best hyperparameters ##################################

print('################# Training final model with best hyperparameters on entire training set #################')

print(f'Best validation metric: {best_val_evaluate} | Params: {best_params}')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30, restore_best_weights=True)
model = create_model(hidden_sizes=best_params['hidden_size'], learning_rates=best_params['learning_rate'], optimizers=optimizers_map[best_params['optimizer']])
history = model.fit(input_seq_train, output_seq_train, validation_split = 0.05, epochs=20, batch_size=64, callbacks=[es], verbose=1)
model.save('best_model_train')

# model = create_model()
# history = model.fit(input_seq_train , output_seq_train, validation_split=0.05, epochs = 20,batch_size=64,verbose=2)

# # #################################### Evaluate model ##############################################################

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