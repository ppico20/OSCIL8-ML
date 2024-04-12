import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
import math as mt
import seaborn as sns
from matplotlib import rc
import matplotlib
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,max_error,explained_variance_score,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import shap
import time
from joblib import dump, load
import sys

# Define the file name for the log
log_file = 'log_HPT_xgb_ak0_tilde.log'

# Function to redirect stdout to the log file
def redirect_stdout_to_log(log_file):
    sys.stdout = open(log_file, 'a', buffering=1)

# Redirect stdout to the log file
redirect_stdout_to_log(log_file)

################################## Plot parameters ##################################

#Plot parameters
color_map = cm.get_cmap('jet', 30)
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ['Computer Modern']})

SMALL_SIZE = 8
MEDIUM_SIZE = 13
BIGGER_SIZE = 14
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

################################################ Reading data ###########################################

start_time = time.time()

print("Beginning file reading ...")

# Reading PCA parameters
u_pod_train = pd.read_csv("u_pod_train.csv", engine='python')

# Reading training and testing data
y_train_original = pd.read_csv("y_train_original.csv", engine='python')
y_train_reconstruct = pd.read_csv("y_train_reconstruct.csv", engine='python')
y_train_compress = pd.read_csv("y_train_compress.csv", engine='python')
y_test_original = pd.read_csv("y_test_original.csv", engine='python')

X_train_original = pd.read_csv("X_train_original.csv", engine='python')
X_test_original = pd.read_csv("X_test_original.csv", engine='python')

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Total time taken to read files is: {elapsed_time} seconds")

################################## Dropping Run_ID and Time from all dataframes ##################################

y_train_original = y_train_original.drop(columns=['Run_ID','Time'])
y_train_reconstruct = y_train_reconstruct.drop(columns=['Run_ID','Time'])
y_train_compress = y_train_compress.drop(columns=['Run_ID','Time'])
y_test_original = y_test_original.drop(columns=['Run_ID','Time'])
X_train_original = X_train_original.drop(columns=['Run_ID','Time'])
X_test_original = X_test_original.drop(columns=['Run_ID','Time'])

################################## Scaling data ##################################

print("Scaling data ...")

scaler = preprocessing.MinMaxScaler()
scaler.fit(X_train_original)

X_train_scaled = scaler.transform(X_train_original)
X_test_scaled = scaler.transform(X_test_original)

X_train_scaled_df = pd.DataFrame(X_train_scaled)
X_test_scaled_df = pd.DataFrame(X_test_scaled)

################################## Hyperparameter tuning ##################################

if not os.path.isfile('xgb_images.joblib'):
    print('Model file does not exist. Performing hyperparameter tuning ...')

    # define hyperparameters
    n_estimators_range = [10, 100, 150]
    max_depth_range = [10, 20, 50]
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
    grid_search.fit(X_train_scaled_df,y_train_compress)

    # get best parameters and best score
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")

    # convert to dataframe
    tunning_xgb = pd.DataFrame(grid_search.cv_results_)
    tunning_xgb.to_csv('tuning_xgb_images.csv')

    end_time = time.time()  
    elapsed_time = end_time - start_time

    print(f"Hyperparameter tunning completed in {elapsed_time:.2f} seconds.")

    xgb = XGBRegressor(n_estimators=grid_search.best_params_.get('n_estimators'),
                        max_depth = grid_search.best_params_.get('max_depth'),random_state=0)
    xgb.fit(X_train_scaled_df,y_train_compress)

else:
    print("Model file found. Hyperparameter tuning not needed")
    xgb = load('xgb_images.joblib')

#xgb = XGBRegressor(n_estimators= 80, max_depth = 15,random_state=0)
#xgb.fit(X_train_scaled_df,y_train_compress)

y_pred_train_compress = xgb.predict(X_train_scaled)
y_pred_test_compress = xgb.predict(X_test_scaled)

y_pred_train_reconstruct = np.dot(u_pod_train,y_pred_train_compress.T).T
y_pred_test_reconstruct = np.dot(u_pod_train,y_pred_test_compress.T).T

#Post-process reconstructed images
threshold = 0.5
y_pred_train_reconstruct = np.where(y_pred_train_reconstruct <= threshold, 0, 1)
y_pred_test_reconstruct = np.where(y_pred_test_reconstruct <= threshold, 0, 1)

print(y_pred_train_reconstruct.shape)
print(y_pred_test_reconstruct.shape)

print('############ Performance metrics in training set ################')
print("Coefficient of determination, r2 = %.5f" % r2_score(y_train_original, y_pred_train_reconstruct))
print("Mean Absolute Error, MAE = %.5f" % mean_absolute_error(y_train_original, y_pred_train_reconstruct))
print("Mean squared error, MSE = %.5f" % mean_squared_error(y_train_original, y_pred_train_reconstruct))
print("Explained Variance Score = %.5f" % explained_variance_score(y_train_original, y_pred_train_reconstruct))

print('############ Performance metrics in testing set ################')
print("Coefficient of determination, r2 = %.5f" % r2_score(y_test_original, y_pred_test_reconstruct))
print("Mean Absolute Error, MAE = %.5f" % mean_absolute_error(y_test_original, y_pred_test_reconstruct))
print("Mean squared error, MSE = %.5f" % mean_squared_error(y_test_original, y_pred_test_reconstruct))
print("Explained Variance Score = %.5f" % explained_variance_score(y_test_original, y_pred_test_reconstruct))

# ##################################### Saving data #################################################

dump(xgb, 'xgb_images.joblib')

pd.DataFrame(y_pred_train_compress).to_csv("y_pred_train_compress.csv", index=False)
pd.DataFrame(y_pred_train_reconstruct).to_csv("y_pred_train_reconstruct.csv", index=False)
pd.DataFrame(y_pred_test_compress).to_csv("y_pred_test_compress.csv", index=False)
pd.DataFrame(y_pred_test_reconstruct).to_csv("y_pred_test_reconstruct.csv", index=False)
X_train_scaled_df.to_csv("X_train_scaled_original.csv",index=False)
X_test_scaled_df.to_csv("X_test_scaled_original.csv",index=False)


