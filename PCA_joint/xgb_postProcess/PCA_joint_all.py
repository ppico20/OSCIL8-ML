#importing the dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, IncrementalPCA
import glob
import re
from sklearn.preprocessing import StandardScaler
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.metrics import mean_squared_error
import os
import time

import sys
from contextlib import redirect_stdout

# Define the file name for the log
log_file = 'log_PCA_joint_all.log'

# Function to redirect stdout to the log file
def redirect_stdout_to_log(log_file):
    sys.stdout = open(log_file, 'a', buffering=1)

# Redirect stdout to the log file
redirect_stdout_to_log(log_file)

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

start_time = time.time()

# ##################################### READING PROCESSED AND PREPARED TRAINING DATA ###################################

print("Beginning file reading ...")

y_train_original = pd.read_csv('y_train_original.csv')

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Total time taken to read training file: {elapsed_time} seconds")
print("The shape of treated dataframe, including Run_ID and Time is:", y_train_original.shape)

# # ##################################### PERFORMING PCA WITH TRAINING SET ONLY ###################################

# REMOVE RUN_ID AND TIME FROM FINAL DATAFRAME TO PERFORM PCA
compiled_df = y_train_original.drop(columns=['Run_ID','Time'])
print(compiled_df)

print("Beginning PCA ...")

u_pod, s_pod, v_pod = np.linalg.svd(compiled_df.T, full_matrices=False)

n_components = 50
u_pod = u_pod[:,:n_components]
field_compress_PCA = np.dot(u_pod.T,compiled_df.T)
field_reconstruct_PCA = np.dot(u_pod,field_compress_PCA).T

print("Beginning image postprocessing ...")
# Post-process reconstructed images
threshold = 0.5
field_reconstruct_PCA = np.where(field_reconstruct_PCA <= threshold, 0, 1)

print(field_compress_PCA.shape)
print(field_reconstruct_PCA.shape)

field_compress_PCA_df = pd.DataFrame(field_compress_PCA.T)
field_compress_PCA_df[['Run_ID', 'Time']] = y_train_original[['Run_ID', 'Time']]

field_reconstruct_PCA_df = pd.DataFrame(field_reconstruct_PCA)
field_reconstruct_PCA_df[['Run_ID', 'Time']] = y_train_original[['Run_ID', 'Time']]

field_compress_PCA_df.to_csv("y_train_compress.csv", index = False)
field_reconstruct_PCA_df.to_csv("y_train_reconstruct.csv", index = False)

u_pod_df = pd.DataFrame(u_pod)
u_pod_df.to_csv("u_pod_train.csv", index = False)

s_pod_df = pd.DataFrame(s_pod)
s_pod_df.to_csv("s_pod_train.csv", index = False)

# ##################################### Performance metrics of PCA ###################################

print("Beginning metrics calculation ...")
# Initialize a list to store the RMSE for each row
rmse_list = []
rrmse_list = []
ssim_list = []

# Iterate over each row of field_reconstruct_PCA
for index, row in enumerate(field_reconstruct_PCA):
    # Extract the corresponding row from compiled_df
    original_row = compiled_df.iloc[index, :].values
    
    # Compute the RMSE between the original and reconstructed rows
    rmse = np.sqrt(mean_squared_error(original_row, row))
    rrmse = rmse / (original_row.max() - original_row.min())
    ssim_score, _ = compare_ssim(original_row, row, full=True,data_range=1)
    
    # Append the RMSE to the list
    rmse_list.append(rmse)
    rrmse_list.append(rrmse)
    ssim_list.append(ssim_score)

# Compute the average RMSE
average_rmse = np.mean(rmse_list)
average_rrmse = np.mean(rrmse_list)
average_ssim = np.mean(ssim_list)

print('############ PCA PERFORMANCE METRICS WITH IMAGE POSTPROCESSING ################')
print("Average RMSE:", average_rmse)
print("Max RMSE:", max(rmse_list))
print("Average RRMSE:", average_rrmse)
print("Max RRMSE:", max(rrmse_list))
print("Average SSIM:", average_ssim)
print("Min SSIM:", min(ssim_list))

# Calculate the total sum of squared singular values
total_variance = np.sum(s_pod ** 2)

# Calculate the percentage of variance captured by each component
variance_explained = (s_pod ** 2) / total_variance * 100

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(variance_explained)

cumulative_variance_df = pd.DataFrame(cumulative_variance)
cumulative_variance_df.to_csv("cumulative_variance_train.csv", index = False)

################################## Figures #############################

plt.figure(1,figsize=[6,5])
ax1 = plt.subplot()
plt.setp(ax1.spines.values(), linewidth=2.0)
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(1.7)  # change width
plt.plot(s_pod[:150],lw=2)
plt.ylabel('Singular values')
plt.xlabel('Truncation parameter')
plt.grid(color='k', linestyle=':', linewidth=0.1)
plt.tight_layout()
plt.savefig("singular_values.png", dpi=2000)


random_train = np.random.choice(len(compiled_df), size=8, replace=False)

fig, axs = plt.subplots(3, 8, figsize=(18, 6), num = 2)
for i in range(8):
    axs[0, i].imshow(compiled_df.iloc[random_train[i],:].to_numpy().reshape(257,257), interpolation='none', vmin=0,vmax=1,cmap='viridis')
    axs[0, i].axis('on')
    axs[0, i].set_xticks([])
    axs[0, i].set_yticks([])
    axs[0, i].set_title(f'{random_train[i]}')
for i in range(8):
    axs[1, i].imshow(field_reconstruct_PCA[random_train[i],:].reshape(257,257), interpolation='none', vmin=0,vmax=1,cmap='viridis')
    axs[1, i].axis('on')
    axs[1, i].set_xticks([])
    axs[1, i].set_yticks([])
    axs[1, i].set_title(f'{random_train[i]}')
for i in range(8):
    difference_image = compiled_df.iloc[random_train[i],:].to_numpy().reshape(257,257) - field_reconstruct_PCA[random_train[i],:].reshape(257,257)
    axs[2, i].imshow(difference_image, interpolation='none', vmin=-1,vmax=1,cmap='viridis')
    axs[2, i].axis('on')
    axs[2, i].set_xticks([])
    axs[2, i].set_yticks([])
    axs[2, i].set_title(f'Difference {random_train[i]}')
plt.tight_layout()
plt.savefig("sample_train_images.png", dpi=2000, bbox_inches='tight')


# print('original ##############################################################################')
# plt.figure(2)
# plt.imshow(compiled_df.iloc[-1,:].to_numpy().reshape(257,257), interpolation="none",vmin=0,vmax=1)
# plt.savefig("sample_y_train_original.png", dpi=2000)

# print('reconstruction ##############################################################################')
# plt.figure(3)
# plt.imshow(field_reconstruct_PCA[-1,:].reshape(257,257), interpolation="none",vmin=0,vmax=1)
# plt.savefig("sample_y_train_reconstruct.png", dpi=2000)

# print('difference ##############################################################################')
# plt.figure(4)
# plt.imshow(compiled_df.iloc[-1,:].to_numpy().reshape(257,257) - field_reconstruct_PCA[-1,:].reshape(257,257), interpolation="none",vmin=-1,vmax=1)
# plt.savefig("diff_sample_y_train.png", dpi=2000)

plt.figure(5,figsize=[6,5])
ax1 = plt.subplot()
plt.setp(ax1.spines.values(), linewidth=2.0)
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(1.7)  # change width
plt.plot(cumulative_variance,lw = 2)
plt.ylabel('Cumulative Explained variance (\%)')
plt.xlabel('Principal components')
plt.title('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance explained by component')
plt.axhline(y=95, color="r", linestyle="--",lw=1)
plt.grid(color='k', linestyle=':', linewidth=0.1)
plt.tight_layout()
plt.savefig("explained_variance.png", dpi=2000)

plt.show()

