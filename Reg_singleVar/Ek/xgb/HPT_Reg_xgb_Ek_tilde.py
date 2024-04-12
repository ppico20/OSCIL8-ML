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

# Define the file name for the log
log_file = 'log_HPT_Reg_xgb_Ek_tilde.log'

# Function to redirect stdout to the log file
def redirect_stdout_to_log(log_file):
    sys.stdout = open(log_file, 'a', buffering=1)

# Redirect stdout to the log file
redirect_stdout_to_log(log_file)

################################## Reading csv ##################################

dataset = pd.read_csv("../../../compiled_data/data_params_HST.csv", engine='python')

data = dataset[['Time_tilde','epsilon','rho_r','mu_r','La_l','Bo_l']]
columns = ['Time_tilde','epsilon','rho_r','mu_r','La_l','Bo_l']

# Selecting labels
label = np.log10(dataset[['Ek_tilde']])

# ################################## Train/test split ##################################

X_train, X_test, y_train, y_test = train_test_split(data, label, random_state=0)

# ################################## Scaling data ##################################

scaler = preprocessing.MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled)
X_test_scaled_df = pd.DataFrame(X_test_scaled)
X_train_scaled_df.columns = X_train.columns
X_test_scaled_df.columns = X_test.columns

print(X_train_scaled_df.mean(axis=0))
print(X_train_scaled_df.var(axis=0))

# check their shape
print("train:", X_train_scaled_df.shape)
print("test:", X_test_scaled.shape)

################################## Hyperparameter tuning ##################################

if not os.path.isfile('xgb_Ek_tilde.joblib'):
    print('Model file does not exist. Performing hyperparameter tuning ...')

    #define hyperparameters
    n_estimators_range = range(50, 200, 4)
    max_depth_range = range(1, 30, 1)
    eta_range = [0.01, 0.1, 0.3]
    gamma_range = [0, 10, 20]
    
    start_time = time.time()

    param_grid = {
        'n_estimators': n_estimators_range,
        'max_depth': max_depth_range,
        'eta': eta_range,
        # 'gamma': gamma_range
    }

    # create classifier
    xgb = XGBRegressor(random_state=0)

    # create grid search with multiple scoring metrics
    scoring = {'r2': 'r2','neg_mean_squared_error':'neg_mean_squared_error'}

    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, n_jobs=-1, scoring=scoring, refit='r2', return_train_score=True)

    # train on grid
    grid_search.fit(X_train_scaled, y_train.iloc[:,0])

    print(y_train)

    # get best parameters and best score
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")

    # convert to dataframe
    tunning_xgb = pd.DataFrame(grid_search.cv_results_)
    tunning_xgb.to_csv('tuning_xgb_Ek_tilde.csv')

    end_time = time.time()  
    elapsed_time = end_time - start_time

    print(f"Hyperparameter tunning completed in {elapsed_time:.2f} seconds.")

    xgb = XGBRegressor(n_estimators=grid_search.best_params_.get('n_estimators'),
                        max_depth = grid_search.best_params_.get('max_depth'),random_state=0)
    xgb.fit(X_train_scaled_df,y_train)

else:
    print("Model file found. Hyperparameter tuning not needed")
    xgb = load('xgb_Ek_tilde.joblib')

#xgb = XGBRegressor(n_estimators= 118, max_depth = 6, eta = 0.3, gamma = 0, random_state=0)

y_pred_train = xgb.predict(X_train_scaled_df)
y_pred_test = xgb.predict(X_test_scaled_df)

# #################################### SAVE MODEL ##############################################################

print('############ Performance metrics in training set ################')
print("Coefficient of determination, r2 = %.5f" % r2_score(y_train, y_pred_train))
print("Mean Absolute Error, MAE = %.5f" % mean_absolute_error(y_train, y_pred_train))
print("Mean squared error, MSE = %.5f" % mean_squared_error(y_train, y_pred_train))
print("Max Error = %.5f" % max_error(y_train, y_pred_train))
print("Explained Variance Score = %.5f" % explained_variance_score(y_train, y_pred_train))

print('############ Performance metrics in testing set ################')
print("Coefficient of determination, r2 = %.5f" % r2_score(y_test, y_pred_test))
print("Mean Absolute Error, MAE = %.5f" % mean_absolute_error(y_test, y_pred_test))
print("Mean squared error, MSE = %.5f" % mean_squared_error(y_test, y_pred_test))
print("Max Error = %.5f" % max_error(y_test, y_pred_test))
print("Explained Variance Score = %.5f" % explained_variance_score(y_test, y_pred_test))

# ##################################### Saving data #################################################

dump(xgb, 'xgb_Ek_tilde.joblib')

pd.DataFrame(X_train).to_csv("X_train.csv",index=False)
pd.DataFrame(X_test).to_csv("X_test.csv",index=False)
X_train_scaled_df.to_csv("X_train_scaled.csv",index=False)
X_test_scaled_df.to_csv("X_test_scaled.csv",index=False)
pd.DataFrame(y_train).to_csv("y_train.csv",index=False)
pd.DataFrame(y_test).to_csv("y_test.csv",index=False)

pd.DataFrame(y_pred_train).to_csv("y_pred_train.csv",index=False)
pd.DataFrame(y_pred_test).to_csv("y_pred_test.csv",index=False)