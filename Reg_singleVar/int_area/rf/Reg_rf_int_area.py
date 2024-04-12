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
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,max_error,explained_variance_score,r2_score
from sklearn.model_selection import cross_val_score
import shap
from joblib import dump, load
import logging
import sys
from contextlib import redirect_stdout

# Define the file name for the log
log_file = 'log_Reg_rf_int_area.log'

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

################################## Read trained and tuned model ##################################

rf = RandomForestRegressor()
rf = load('rf_int_area.joblib')

tuning = pd.read_csv('tuning_rf_int_area.csv')

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')

X_train_scaled = pd.read_csv('X_train_scaled.csv')
X_test_scaled = pd.read_csv('X_test_scaled.csv')

y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

y_pred_train = rf.predict(X_train_scaled)
y_pred_test = rf.predict(X_test_scaled)

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

# Feature importance
importance_scores = rf.feature_importances_
print(importance_scores)

# # Shap values
# Create the explainer
explainer = shap.TreeExplainer(rf)
# xgb_shap_values = explainer.shap_values(X_test_scaled)
rf_shap_values = explainer(X_test_scaled)
print(rf_shap_values.shape)

# ################################## Plots ##################################

plt.figure(1,figsize=[8,5])
arr_test_heatmap = tuning.pivot_table(index="param_n_estimators", columns="param_max_depth", values="mean_test_r2",aggfunc='first')
ax1 = sns.heatmap(arr_test_heatmap, vmin=0.6, vmax=1,cbar_kws={'label': r'$R^{2}$'})
plt.setp(ax1.spines.values(), linewidth=1.3)
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(1.7)  # change width
ax1.set(xlabel=r"$Max_{depth}$", ylabel=r"$n_{estimators}$")
# ax1.set_xlim(1, 100)
# ax1.set_ylim(160,10)
plt.tight_layout()
plt.savefig("HPT_map_rf_int_area.png", dpi=2000)

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
plt.scatter(y_train,(y_train.iloc[:,0] - y_pred_train),c=X_train["Time_tilde"], edgecolor='k',cmap='viridis')
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
plt.scatter(y_test,(y_test.iloc[:,0] - y_pred_test),c=X_test["Time_tilde"], edgecolor='k',cmap='viridis')
plt.axhline(y = 0, color = 'k', linestyle = '-')
plt.ylim([-1.5,1.5])
plt.xlabel(r"$\textrm{Data Point}$")
plt.ylabel(r"$\textrm{Residual}$")
plt.grid(color='k', linestyle=':', linewidth=0.1)
plt.colorbar(label=r'$\tilde{t}$')  # Add colorbar with label
ax1.tick_params(direction='in', length=6, width=1)
plt.tight_layout()
plt.savefig("pred_rf_int_area.png", dpi=2000)

plt.figure(3,figsize=[5.5,4])

latex_mapping = {
    'Time_tilde': r'$\tilde{t}$',
    'epsilon': r'$\epsilon$',
    'rho_r': r'$\rho_{r}$',
    'mu_r': r'$\mu_{r}$', 
    'La_l': r'$La_{l}$', 
    'Bo_l': r'$Bo_{l}$',
}

feature_names = X_test_scaled.columns 
latex_feature_names = [latex_mapping[name] for name in feature_names]

ax1 = plt.subplot()
plt.setp(ax1.spines.values(), linewidth=1.3)
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(1.7)  # change width
rects = plt.bar(range(len(importance_scores)), importance_scores, facecolor = 'r', edgecolor = 'k')
ax1.bar_label(rects, padding=0)
plt.xticks(range(len(importance_scores)), latex_feature_names, rotation='horizontal')
plt.ylabel(r'$\textrm{Importance Scores}$')
plt.title('Feature Importance')
ax1.set_ylim(0, 0.65)
plt.tight_layout()
plt.savefig("feat_importance_rf_int_area.png", dpi=2000)

fig = plt.figure(4,figsize=[5.5,4])

feature_names = X_test_scaled.columns 
latex_feature_names = [latex_mapping[name] for name in feature_names]
ax1 = shap.summary_plot(rf_shap_values, X_test_scaled,axis_color='k',feature_names=latex_feature_names,plot_size=(7,5))
fig.savefig("shap_rf_int_area.png", dpi=2000)

plt.show()


