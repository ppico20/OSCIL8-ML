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
from scipy.signal import find_peaks
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,max_error,explained_variance_score,r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import seaborn as sns; sns.set_theme()
import shap


#Plot parameters
color_map = cm.get_cmap('jet', 30)
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ['Computer Modern']})

SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 13
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

################################## Reading csv ##################################
# dataset = pd.read_csv("../compiled_data/data_params_rate_int_area.csv")

dataset = pd.read_csv("../compiled_data/data_params_HST.csv")

# Selecting features
# data = dataset[['Time_tilde','a0','k','sigma_s','rho_l','rho_g','mu_l','mu_g','gravity']]
# columns = ['Time_tilde','a0','k','sigma_s','rho_l','rho_g','mu_l','mu_g','gravity']

data = dataset[['Time_tilde','epsilon','rho_r','mu_r','La_l','Bo_l']]
columns = ['Time_tilde','epsilon','rho_r','mu_r','La_l','Bo_l']

# Selecting labels
label = dataset[['Int_area_tilde']]

################################## Train/test split ##################################

X_train, X_test, y_train, y_test = train_test_split(data, label, random_state=0)

################################## Scaling data ##################################
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled)
X_train_scaled_df.columns = X_train.columns

print(X_train_scaled_df.mean(axis=0))
print(X_train_scaled_df.var(axis=0))

# check their shape
print("train:", X_train_scaled_df.shape)
# print("valid:", X_valid.shape)
print("test:", X_test_scaled.shape)

################################## Random forest regression ##################################

rf = RandomForestRegressor(n_estimators=70, max_depth = 28,random_state=0)
rf.fit(X_train_scaled_df,y_train)
y_pred = rf.predict(X_test_scaled)

print("Coefficient of determination, r2 = %.2f" % r2_score(y_test, y_pred))
print("Mean Absolute Error, MAE = %.2f" % mean_absolute_error(y_test, y_pred))
print("Mean squared error, MSE = %.2f" % mean_squared_error(y_test, y_pred))
print("Max Error = %.2f" % max_error(y_test, y_pred))
print("Explained Variance Score = %.2f" % explained_variance_score(y_test, y_pred))

# Feature importance
importance_scores = rf.feature_importances_
print(importance_scores)

# # Shap values
# # Create the explainer
# explainer = shap.TreeExplainer(rf)
# rf_shap_values = explainer.shap_values(X_test_scaled)


################################## Cross validation/hyperparameter tunning ##################################

# # define hyperparameters
# n_estimators_range = range(1, 100, 5)
# max_depth_range = range(3, 50, 5)

# param_grid = {
#     'n_estimators': n_estimators_range,
#     'max_depth': max_depth_range,
# }

# # create classifier
# rf = RandomForestRegressor(random_state=42)

# # create grid search with multiple scoring metrics
# scoring = {'r2': 'r2','neg_mean_squared_error':'neg_mean_squared_error'}

# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring=scoring, refit='r2', return_train_score=True)

# # train on grid
# grid_search.fit(X_train_scaled, y_train.iloc[:,0])

# # get best parameters and best score (f1)
# print(f"Best parameters: {grid_search.best_params_}")
# print(f"Best score: {grid_search.best_score_}")

# # predict on test set
# y_pred = grid_search.predict(X_test_scaled)

# # convert to dataframe
# results_rf = pd.DataFrame(grid_search.cv_results_)
# print(results_rf.columns)

# results_rf = results_rf[['param_n_estimators', 'param_max_depth', 'mean_test_r2', 'mean_test_neg_mean_squared_error']]

# results_rf.columns = ['param_n_estimators', 'param_max_depth', 'mean_test_r2', 'mean_test_neg_mean_squared_error']

# results_rf.loc[:, 'param_n_estimators'] = results_rf['param_n_estimators'].astype(int)
# results_rf.loc[:, 'param_max_depth'] = results_rf['param_max_depth'].astype(int)

# print(f"Best parameters: {grid_search.best_params_}")
# print(f"Best score: {grid_search.best_score_}")

# sns.set(font_scale = 1)
# arr_test_heatmap = results_rf.pivot(index="param_n_estimators", columns="param_max_depth", values="mean_test_r2")
# ax_test = sns.heatmap(arr_test_heatmap, cmap = "Blues", vmin=0.9, vmax=1)
# print("\naccuracy heatmap:")
# plt.show()

################################## Plots ##################################

plt.figure(1,figsize=[8,5])

ax1 = plt.subplot()
plt.setp(ax1.spines.values(), linewidth=1.3)
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(1.2)  # change width
plt.scatter(y_pred,y_test,c=X_test["Time_tilde"], edgecolor='k')
plt.plot(y_test,y_test,color = 'r',markersize=0)
plt.plot(y_test + 0.025*y_test,y_test,color = 'r',markersize=0)
plt.plot(y_test - 0.025*y_test,y_test,color = 'r',markersize=0)
# plt.axhline(y = 1, color = 'k', linestyle = '-')
# plt.axhline(y = -1, color = 'k', linestyle = '-')
# ax1.legend().set_visible(False)
plt.xlabel(r"Prediction")
plt.ylabel(r"Simulation")
plt.grid(color='k', linestyle=':', linewidth=0.1)
plt.colorbar(label=r'$\tilde{t}$')  # Add colorbar with label
# plt.xlim([0,4])
# plt.ylim([0,4])
# plt.xticks([0,5,10,15,20,25,30])
# # plt.yticks([-1,-0.5,0,0.5,1])
ax1.tick_params(direction='in', length=6, width=1)

plt.figure(2,figsize=[8,5])

ax1 = plt.subplot()
plt.setp(ax1.spines.values(), linewidth=1.3)
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(1.2)  # change width
plt.scatter(y_pred,(y_test.iloc[:,0] - y_pred),c=X_test["Time_tilde"], edgecolor='k')
plt.axhline(y = 0, color = 'k', linestyle = '-')
# plt.axhline(y = -1, color = 'k', linestyle = '-')
# ax1.legend().set_visible(False)
plt.xlabel(r"Prediction")
plt.ylabel(r"Residual")
plt.grid(color='k', linestyle=':', linewidth=0.1)
plt.colorbar(label=r'$\tilde{t}$')  # Add colorbar with label
# plt.xlim([0,4])
# plt.ylim([0,4])
# plt.xticks([0,5,10,15,20,25,30])
# # plt.yticks([-1,-0.5,0,0.5,1])
ax1.tick_params(direction='in', length=6, width=1)


plt.figure(3,figsize=[8,5])

ax1 = plt.subplot()
plt.setp(ax1.spines.values(), linewidth=1.3)
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(1.2)  # change width
feature_names = data.columns 
plt.bar(range(len(importance_scores)), importance_scores)
plt.xticks(range(len(importance_scores)), feature_names, rotation='vertical')
plt.xlabel('Features')
plt.ylabel('Importance Scores')
plt.title('Feature Importance')


# plt.figure(4,figsize=[8,5])

# ax1 = plt.subplot()
# shap.summary_plot(rf_shap_values, X_test_scaled)

plt.show()
