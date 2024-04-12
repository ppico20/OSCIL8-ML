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
log_file = 'log_Reg_xgb_ak0_tilde.log'

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

################################## Read trained and tuned model ##################################

print("Loading tuned model ...")

xgb = XGBRegressor()
xgb = load('xgb_images.joblib')

tuning = pd.read_csv('tuning_xgb_images.csv')

u_pod_train = pd.read_csv("u_pod_train.csv", engine='python')

print("Loading data ...")

X_train_original = pd.read_csv('X_train_original.csv')
X_test_original = pd.read_csv('X_test_original.csv')

X_train_scaled = pd.read_csv('X_train_scaled_original.csv')
X_test_scaled = pd.read_csv('X_test_scaled_original.csv')

y_train_original = pd.read_csv("y_train_original.csv", engine='python')
y_train_reconstruct = pd.read_csv("y_train_reconstruct.csv", engine='python')
y_train_compress = pd.read_csv("y_train_compress.csv", engine='python')
y_test_original = pd.read_csv("y_test_original.csv", engine='python')

################################## Dropping Run_ID and Time from all dataframes ##################################

y_train_original = y_train_original.drop(columns=['Run_ID','Time'])
y_train_reconstruct = y_train_reconstruct.drop(columns=['Run_ID','Time'])
y_train_compress = y_train_compress.drop(columns=['Run_ID','Time'])
y_test_original = y_test_original.drop(columns=['Run_ID','Time'])
X_train_original = X_train_original.drop(columns=['Run_ID','Time'])
X_test_original = X_test_original.drop(columns=['Run_ID','Time'])

######################### Making predictions ##################################

print("Predictions using tuned model ...")

y_pred_train_compress = xgb.predict(X_train_scaled)
y_pred_test_compress = xgb.predict(X_test_scaled)

y_pred_train_reconstruct = np.dot(u_pod_train,y_pred_train_compress.T).T
y_pred_test_reconstruct = np.dot(u_pod_train,y_pred_test_compress.T).T

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

# Feature importance
importance_scores = xgb.feature_importances_
print(importance_scores)

# # Shap values
#explainer = shap.TreeExplainer(xgb)
#xgb_shap_values = explainer(X_test_scaled)

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
plt.savefig("HPT_map_xgb_images.png", dpi=2000)

random_train = np.random.choice(len(y_pred_train_reconstruct), size=8, replace=False)

fig, axs = plt.subplots(3, 8, figsize=(18, 6),num = 2)
for i in range(8):
    axs[0, i].imshow(y_train_original.iloc[random_train[i],:].to_numpy().reshape(257,257), interpolation='none', vmin=0,vmax=1,cmap='viridis')
    axs[0, i].axis('on')
    axs[0, i].set_xticks([])
    axs[0, i].set_yticks([])
    axs[0, i].set_title(f'{random_train[i]}')
for i in range(8):
    axs[1, i].imshow(y_pred_train_reconstruct[random_train[i],:].reshape(257,257), interpolation='none', vmin=0,vmax=1,cmap='viridis')
    axs[1, i].axis('on')
    axs[1, i].set_xticks([])
    axs[1, i].set_yticks([])
    axs[1, i].set_title(f'{random_train[i]}')
for i in range(8):
    difference_image = y_train_original.iloc[random_train[i],:].to_numpy().reshape(257,257) - y_pred_train_reconstruct[random_train[i],:].reshape(257,257)
    axs[2, i].imshow(difference_image, interpolation='none', vmin=-1,vmax=1,cmap='viridis')
    axs[2, i].axis('on')
    axs[2, i].set_xticks([])
    axs[2, i].set_yticks([])
    axs[2, i].set_title(f'Difference {random_train[i]}')
plt.tight_layout()
plt.savefig("sample_pred_train_images.png", dpi=2000, bbox_inches='tight')

random_test = np.random.choice(len(y_pred_test_reconstruct), size=8, replace=False)

fig, axs = plt.subplots(3, 8, figsize=(18, 6), num=3)
for i in range(8):
    axs[0, i].imshow(y_test_original.iloc[random_test[i],:].to_numpy().reshape(257,257), interpolation='none', vmin=0,vmax=1,cmap='viridis')
    axs[0, i].axis('on')
    axs[0, i].set_xticks([])
    axs[0, i].set_yticks([])
    axs[0, i].set_title(f'{random_test[i]}')
for i in range(8):
    axs[1, i].imshow(y_pred_test_reconstruct[random_test[i],:].reshape(257,257), interpolation='none', vmin=0,vmax=1,cmap='viridis')
    axs[1, i].axis('on')
    axs[1, i].set_xticks([])
    axs[1, i].set_yticks([])
    axs[1, i].set_title(f'{random_test[i]}')
for i in range(8):
    difference_image = y_test_original.iloc[random_test[i],:].to_numpy().reshape(257,257) - y_pred_test_reconstruct[random_test[i],:].reshape(257,257)
    axs[2, i].imshow(difference_image, interpolation='none', vmin=-1,vmax=1,cmap='viridis')
    axs[2, i].axis('on')
    axs[2, i].set_xticks([])
    axs[2, i].set_yticks([])
    axs[2, i].set_title(f'Difference {random_test[i]}')
plt.tight_layout()
plt.savefig("sample_pred_test_images.png", dpi=2000, bbox_inches='tight')

#plt.figure(4,figsize=[5.5,4])

#latex_mapping = {
#    'Time_tilde': r'$\tilde{t}$',
#    'epsilon': r'$\epsilon$',
#    'rho_r': r'$\rho_{r}$',
#    'mu_r': r'$\mu_{r}$', 
#    'La_l': r'$La_{l}$', 
#    'Bo_l': r'$Bo_{l}$',
#}

#feature_names = X_train_original.columns 
#latex_feature_names = [latex_mapping[name] for name in feature_names]

#ax1 = plt.subplot()
#plt.setp(ax1.spines.values(), linewidth=1.3)
#for axis in ['top', 'bottom', 'left', 'right']:
#    ax1.spines[axis].set_linewidth(1.7)  # change width
#rects = plt.bar(range(len(importance_scores)), importance_scores, facecolor = 'r', edgecolor = 'k')
#ax1.bar_label(rects, padding=0)
#plt.xticks(range(len(importance_scores)), latex_feature_names, rotation='horizontal')
#plt.ylabel(r'$\textrm{Importance Scores}$')
#plt.title('Feature Importance')
#ax1.set_ylim(0, 0.65)
#plt.tight_layout()
#plt.savefig("feat_importance_xgb_images.png", dpi=2000)

# fig = plt.figure(5,figsize=[5.5,4])

# feature_names = X_test_original.columns
# latex_feature_names = [latex_mapping[name] for name in feature_names]
# ax1 = shap.summary_plot(xgb_shap_values, X_test_scaled,axis_color='k',feature_names=latex_feature_names,plot_size=(7,5))
# fig.savefig("shap_xgb_images.png", dpi=2000)

plt.show()
