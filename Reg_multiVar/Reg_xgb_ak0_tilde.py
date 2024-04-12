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
from sklearn.multioutput import MultiOutputRegressor
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
data = dataset[['Time_tilde','a0','k','sigma_s','rho_l','rho_g','mu_l','mu_g','gravity']]
columns = ['Time_tilde','a0','k','sigma_s','rho_l','rho_g','mu_l','mu_g','gravity']

# data = dataset[['Time_tilde','epsilon','rho_r','mu_r','La_l','Bo_l']]
# columns = ['Time_tilde','epsilon','rho_r','mu_r','La_l','Bo_l']

# Selecting labels
label = dataset[['ak0_tilde','ak1_tilde','ak2_tilde','ak3_tilde']]

################################## Train/test split ##################################

X_train, X_test, y_train, y_test = train_test_split(data, label, random_state=0)

################################## Scaling data ##################################
# scaler = StandardScaler()
# scaler.fit(X_train)

scaler = preprocessing.MinMaxScaler()
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

################################## XGBoost regression ##################################

xgb = XGBRegressor(n_estimators=70, max_depth = 28,random_state=0,tree_method="hist")
multioutput_regressor = MultiOutputRegressor(xgb)
multioutput_regressor.fit(X_train_scaled_df,y_train)
y_pred = multioutput_regressor.predict(X_test_scaled)

y_pred_train = multioutput_regressor.predict(X_train_scaled)


print("Coefficient of determination, r2 = %.5f" % r2_score(y_test, y_pred))
print("Mean Absolute Error, MAE = %.5f" % mean_absolute_error(y_test, y_pred))
print("Mean squared error, MSE = %.5f" % mean_squared_error(y_test, y_pred))
# print("Max Error = %.5f" % max_error(y_test, y_pred))
print("Explained Variance Score = %.5f" % explained_variance_score(y_test, y_pred))

# Feature importance
# importance_scores = multioutput_regressor.feature_importances_
# print(importance_scores)

# # Shap values
# # Create the explainer
# explainer = shap.TreeExplainer(multioutput_regressor)
# multioutput_regressor_shap_values = explainer.shap_values(X_test_scaled)


################################## Plots ##################################

plt.figure(1,figsize=[8,5])

ax1 = plt.subplot()
plt.setp(ax1.spines.values(), linewidth=1.3)
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(1.2)  # change width
plt.scatter(y_pred,y_test, edgecolor='k')
plt.plot(y_test,y_test,color = 'r',markersize=0)
# plt.plot(y_test + 0.5*y_test,y_test,color = 'r',markersize=0)
# plt.plot(y_test - 0.5*y_test,y_test,color = 'r',markersize=0)
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

# plt.figure(2,figsize=[8,5])

# ax1 = plt.subplot()
# plt.setp(ax1.spines.values(), linewidth=1.3)
# for axis in ['top', 'bottom', 'left', 'right']:
#     ax1.spines[axis].set_linewidth(1.2)  # change width
# plt.scatter(y_pred,(y_test.iloc[:,0] - y_pred),c=X_test["Time_tilde"], edgecolor='k')
# plt.axhline(y = 0, color = 'k', linestyle = '-')
# # plt.axhline(y = -1, color = 'k', linestyle = '-')
# # ax1.legend().set_visible(False)
# plt.xlabel(r"Prediction")
# plt.ylabel(r"Residual")
# plt.grid(color='k', linestyle=':', linewidth=0.1)
# plt.colorbar(label=r'$\tilde{t}$')  # Add colorbar with label
# # plt.xlim([0,4])
# # plt.ylim([0,4])
# # plt.xticks([0,5,10,15,20,25,30])
# # # plt.yticks([-1,-0.5,0,0.5,1])
# ax1.tick_params(direction='in', length=6, width=1)

# plt.figure(3,figsize=[8,5])

# ax1 = plt.subplot()
# plt.setp(ax1.spines.values(), linewidth=1.3)
# for axis in ['top', 'bottom', 'left', 'right']:
#     ax1.spines[axis].set_linewidth(1.2)  # change width
# feature_names = data.columns 
# plt.bar(range(len(importance_scores)), importance_scores)
# plt.xticks(range(len(importance_scores)), feature_names, rotation='vertical')
# plt.xlabel('Features')
# plt.ylabel('Importance Scores')
# plt.title('Feature Importance')


# plt.figure(4,figsize=[8,5])

# ax1 = plt.subplot()
# shap.summary_plot(xgb_shap_values, X_test_scaled)

plt.show()
