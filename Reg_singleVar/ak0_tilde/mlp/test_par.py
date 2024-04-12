# Import modules
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
import math as mt
import seaborn as sns
from matplotlib import rc
import matplotlib
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
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
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras import optimizers
from sklearn.model_selection import StratifiedKFold
from tensorflow_addons.metrics import RSquare
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from tensorflow.errors import InvalidArgumentError
from keras.models import load_model
import json
import itertools
import csv
import time
import multiprocessing

# Define the file name for the log
log_file = 'log_HPT_Reg_mlp_ak0_tilde.log'

# Function to redirect stdout to the log file
def redirect_stdout_to_log(log_file):
    sys.stdout = open(log_file, 'a', buffering=1)

# Redirect stdout to the log file
redirect_stdout_to_log(log_file)

################################## Plot parameters ##################################

color_map = cm.get_cmap('jet', 30)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ['Computer Modern']})

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 15
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE + 2)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


################################## Reading csv ##################################

dataset = pd.read_csv("../../../compiled_data/data_params_HST.csv", engine='python')

data = dataset[['Time_tilde','epsilon','rho_r','mu_r','La_l','Bo_l']]
columns = ['Time_tilde','epsilon','rho_r','mu_r','La_l','Bo_l']

# Selecting labels
label = dataset[['ak0_tilde']]

################################## Train/test split ##################################

X_train, X_test, y_train, y_test = train_test_split(data, label, random_state=42)

################################## Scaling data ##################################

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
print("test:", X_test_scaled.shape,"\n")

################################## Creating NN model ##################################

def create_model(activation='relu', learning_rates=0.001, dropout_rate=0.2, l2_rate=0.0001, optimizer=optimizers.Adam):
    """
    This function creates the general structure of the NN with default hyperparameter values
    """
    model = Sequential()
    model.add(Dense(256, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l2_rate), input_shape=(len(columns),)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l2_rate)))
    model.add(Dense(64, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l2_rate)))
    model.add(Dense(32, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l2_rate)))
    model.add(Dense(16, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l2_rate)))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer=optimizer(learning_rate=learning_rates), metrics=[RSquare()])
    return model

################################## Function to train our NN for each set of hyperparameters ##################################

def train_model_HPT(activation, optimizer, learning_rate, l2_rate, dropout_rate, X_train, y_train, X_val, y_val):
    """
    This function is tasked with training the NN on each parameter combination for hyperparameter tunning (on each fold).
    It uses an eary stop on the loss function (mse) of the validation set.
    Returns the trained model, the performance metric on the validation data, and the training history.
    """

    model = create_model(activation=activation, learning_rates=learning_rate, l2_rate=l2_rate, dropout_rate=dropout_rate, optimizer=optimizer)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30, restore_best_weights=True)
    hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=600, callbacks=[es], verbose=0)
    val_evaluate = model.evaluate(X_val, y_val, verbose=0)[1]
    return model, hist, val_evaluate

def train_model_HPT_worker(params):
    """
    Worker function for training a single model with specific hyperparameters.
    """
    activation, optimizer_key, learning_rate, l2, dropout_rate, X_train, y_train, X_val, y_val = params
    return train_model_HPT(activation, optimizers_map[optimizer_key], learning_rate, l2, dropout_rate, X_train, y_train, X_val, y_val)


def tune_hyperparameters_parallel(param_combinations, X_train, y_train, X_val, y_val, kfold):
    """
    Perform hyperparameter tuning in parallel.
    """
    pool = multiprocessing.Pool()
    results = []

    for params in param_combinations:
        results.append(pool.apply_async(train_model_HPT_worker, ((params[0], params[1], params[2], params[3], params[4], X_train[train_idx], y_train[train_idx], X_train[val_idx], y_train[val_idx]) for train_idx, val_idx in kfold.split(X_train, y_train))))

    pool.close()
    pool.join()

    # Retrieve results
    tuning_results = [result.get() for result in results]

    return tuning_results


################################## Hyperparameter tuning function with k-fold ##################################
def tune_hyperparameters(activation_functions, optimizers_map, learning_rates, l2_rates, dropout_rates, X_train, y_train, kfold):
    """
    This function performs hyperparameter tunning. It goes over each list of hyperparameters (activation functions, optimisers,
    etc) and creates a k-fold cross validation for each parameter combination. It returns the model that produced the best average
    r2 (over the k folds), its history, and the best parameters)
    """

    fieldnames = ['Index', 'Activation', 'Optimizer', 'Learning Rate', 'L2 Rate', 'Dropout Rate', 'Mean Val Evaluate']
    csvfile = open(output_csv_file, 'w', newline='')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    best_val_evaluate = float('-inf')
    best_params = {}
    tuning_results = []
    param_combinations = itertools.product(activation_functions, optimizers_map.keys(), learning_rates, l2_rates, dropout_rates)
    
    for idx, (activation, optimizer_key, learning_rate, l2, dropout_rate) in enumerate(param_combinations):
        print('############################# Testing a new configuration #################################')
        print(f'Parameter Configuration Index: {idx}')
        print(f'Current settings: activation={activation}, optimizer={optimizer_key}, learning rate={learning_rate}, l2_rate={l2}, dropout rate={dropout_rate} \n')
        val_evaluate_list = []
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
            print(f'Currently on fold {fold_idx}')
            try:
                model, hist, val_evaluate = train_model_HPT(activation, optimizers_map[optimizer_key], learning_rate, l2, dropout_rate, X_train[train_idx], y_train[train_idx], X_train[val_idx], y_train[val_idx])
                val_evaluate_list.append(val_evaluate)
            except Exception as e:
                print(f"Error occurred for params: activation={activation}, optimizer={optimizer_key}, learning_rate={learning_rate}, l2_rate={l2}, dropout rate={dropout_rate}")
                print(str(e))
                continue
        mean_val_evaluate = np.mean(val_evaluate_list)
        print(f'Average metric value in this configuration (validation data): {mean_val_evaluate}')
        tuning_results.append((activation, optimizer_key, learning_rate, l2, dropout_rate, mean_val_evaluate))
        row_data = {'Index': idx, 'Activation': activation, 'Optimizer': optimizer_key, 'Learning Rate': learning_rate, 'L2 Rate': l2, 'Dropout Rate': dropout_rate, 'Mean Val Evaluate': mean_val_evaluate}
        writer.writerow(row_data)
        csvfile.flush()
        print('############################################################################################# \n')
        
        if mean_val_evaluate > best_val_evaluate:
            best_val_evaluate = mean_val_evaluate
            best_params = {'activation': activation, 'optimizer': optimizer_key, 'learning_rate': learning_rate, 'l2_rate': l2, 'dropout_rate': dropout_rate}
            print(f'Best validation metric value: {best_val_evaluate} | Params: {best_params} \n')
            # Save best model and its history
            model.save(best_model_kfold_path)
            history = hist
    
    csvfile.close()
    return model, history, best_params, best_val_evaluate, tuning_results

################################## Declaration of hyperparameters ranges to explore ##################################

# define hyperparameters
activation_functions = ['relu','tanh','sigmoid']
optimizers_map = {'adam': optimizers.Adam,'sgd': optimizers.SGD, 'rmsprop': optimizers.RMSprop}
learning_rates = [0.002, 0.01]
l2_rates = [0, 0.0001, 0.001]
dropout_rates = [0,0.1]

# Kfold definition
kfold = KFold(n_splits=2, shuffle=True)

################################## Performing hyperparameter tunning ##################################

if __name__ == "__main__":
    start_time = time.time()
    output_csv_file = 'tuning_mlp_ak0_tilde.csv'
    best_model_kfold_path = 'best_model_train_kfold'
    y_train_np = y_train.values
    param_combinations = list(itertools.product(activation_functions, optimizers_map.keys(), learning_rates, l2_rates, dropout_rates))
    tuning_results = tune_hyperparameters_parallel(param_combinations, X_train_scaled, y_train_np, X_train_scaled, y_train_np, kfold)
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
    model = create_model(activation=best_params['activation'], learning_rates=best_params['learning_rate'], l2_rate=best_params['l2_rate'], dropout_rate=best_params['dropout_rate'], optimizer=optimizers_map[best_params['optimizer']])
    history = model.fit(X_train_scaled_df, y_train, validation_split = 0.2, epochs=600, callbacks=[es], verbose=1)
    model.save('best_model_train')

    ################################## Model predictions ##################################

    y_pred_train = model.predict(X_train_scaled_df)
    y_pred_test = model.predict(X_test_scaled_df)

    model.summary()

    explainer = shap.DeepExplainer(model,X_train_scaled_df.values[:1000])
    nn_shap_values = explainer.shap_values(X_test_scaled_df.values[:1000])

    # split negative and positive shap values
    nn_shap_values_positive = nn_shap_values[0]
    nn_shap_values_negative = -nn_shap_values[0]

    # create a list，store shap values for positive and negative class
    nn_shap_values_list = [nn_shap_values_negative, nn_shap_values_positive]

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


    ################################################## Plots ##################################################
    latex_mapping = {
        'Time_tilde': r'$\tilde{t}$',
        'epsilon': r'$\epsilon$',
        'rho_r': r'$\rho_{r}$',
        'mu_r': r'$\mu_{r}$', 
        'La_l': r'$La_{l}$', 
        'Bo_l': r'$Bo_{l}$',
    }

    plt.figure(1,figsize=[8,4])
    ax1 = plt.subplot(111)
    plt.setp(ax1.spines.values(), linewidth=1.3)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(1.7)  # change width
    plt.plot(pd.DataFrame(history.history['loss']),'o', markersize=0,markeredgewidth=1.0, markeredgecolor='k',ls=('-'),lw=1.7,color='r',label="Train")
    plt.plot(pd.DataFrame(history.history['val_loss']),'o', markersize=0,markeredgewidth=1.0, markeredgecolor='k',ls=('-'),lw=1.7,color='b',label='Validation')
    plt.xlabel(r"$\textrm{Epoch}$")
    plt.ylabel("$MSE$")
    leg = plt.legend(prop={'size': 11}, loc='best', ncol=1,fancybox=False, framealpha=1, shadow=False)
    leg.get_frame().set_linewidth(1.2)
    leg.get_frame().set_edgecolor('k')
    plt.grid(color='k', linestyle=':', linewidth=0.1)
    ax1.tick_params(bottom=True, top=True, left=True, right=True,axis='both',direction='in', length=3, width=1)
    plt.tight_layout()
    plt.savefig("history_mlp_ak0_tilde.png", dpi=2000)

    plt.figure(2,figsize=[12,8])

    ax1 = plt.subplot(221)
    plt.title('Training data')
    plt.setp(ax1.spines.values(), linewidth=1.3)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(1.7)  # change width
    plt.scatter(y_pred_train,y_train,c=X_train["Time_tilde"], edgecolor='k',cmap='viridis')
    plt.plot(y_train,y_train,color = 'k',markersize=0)
    # plt.plot(y_test + 0.05*y_test,y_test,color = 'r',markersize=0)
    # plt.plot(y_test - 0.05*y_test,y_test,color = 'r',markersize=0)
    plt.xlabel(r"$\textrm{Prediction}$")
    plt.ylabel(r"$\textrm{Simulation}$")
    plt.grid(color='k', linestyle=':', linewidth=0.1)
    plt.colorbar(label=r'$\tilde{t}$')  # Add colorbar with label
    ax1.tick_params(direction='in', length=6, width=1)
    plt.tight_layout()

    ax1 = plt.subplot(222)
    plt.title('Testing data')
    plt.setp(ax1.spines.values(), linewidth=1.3)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(1.7)  # change width
    plt.scatter(y_pred_test,y_test,c=X_test["Time_tilde"], edgecolor='k',cmap='viridis')
    plt.plot(y_test,y_test,color = 'k',markersize=0)
    # plt.plot(y_test + 0.2*y_test,y_test,color = 'r',markersize=0)
    # plt.plot(y_test - 0.2*y_test,y_test,color = 'r',markersize=0)
    plt.xlabel(r"$\textrm{Prediction}$")
    plt.ylabel(r"$\textrm{Simulation}$")
    plt.grid(color='k', linestyle=':', linewidth=0.1)
    plt.colorbar(label=r'$\tilde{t}$')
    ax1.tick_params(direction='in', length=6, width=1)
    plt.tight_layout()

    ax1 = plt.subplot(223)
    plt.setp(ax1.spines.values(), linewidth=1.3)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(1.7)  # change width
    plt.scatter(y_train,(y_train - y_pred_train),c=X_train["Time_tilde"], edgecolor='k',cmap='viridis')
    plt.axhline(y = 0, color = 'k', linestyle = '-')
    plt.ylim([-1.5,1.5])
    plt.xlabel(r"$\textrm{Data Point}$")
    plt.ylabel(r"$\textrm{Residual}$")
    plt.grid(color='k', linestyle=':', linewidth=0.1)
    plt.colorbar(label=r'$\tilde{t}$')  # Add colorbar with label
    ax1.tick_params(direction='in', length=6, width=1)
    plt.tight_layout()

    ax1 = plt.subplot(224)
    plt.setp(ax1.spines.values(), linewidth=1.3)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(1.7)  # change width
    plt.scatter(y_test,(y_test - y_pred_test),c=X_test["Time_tilde"], edgecolor='k',cmap='viridis')
    plt.axhline(y = 0, color = 'k', linestyle = '-')
    plt.ylim([-1.5,1.5])
    plt.xlabel(r"$\textrm{Data Point}$")
    plt.ylabel(r"$\textrm{Residual}$")
    plt.grid(color='k', linestyle=':', linewidth=0.1)
    plt.colorbar(label=r'$\tilde{t}$')  # Add colorbar with label
    ax1.tick_params(direction='in', length=6, width=1)
    plt.tight_layout()
    plt.savefig("pred_mlp_ak0_tilde.png", dpi=2000)

    fig = plt.figure(4,figsize=[5.5,4])

    feature_names = X_test_scaled_df.columns 
    latex_feature_names = [latex_mapping[name] for name in feature_names]
    ax1 = shap.summary_plot(nn_shap_values_list[1], X_test_scaled[:1000],axis_color='k',feature_names=latex_feature_names)
    fig.savefig("shap_mlp_ak0_tilde.png", dpi=2000)

    plt.show()

    # ##################################### Saving data #################################################

    pd.DataFrame(X_train).to_csv("X_train.csv",index=False)
    pd.DataFrame(X_test).to_csv("X_test.csv",index=False)
    X_train_scaled_df.to_csv("X_train_scaled.csv",index=False)
    X_test_scaled_df.to_csv("X_test_scaled.csv",index=False)
    pd.DataFrame(y_train).to_csv("y_train.csv",index=False)
    pd.DataFrame(y_test).to_csv("y_test.csv",index=False)

