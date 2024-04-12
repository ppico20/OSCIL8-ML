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

#Plot parameters
color_map = cm.get_cmap('jet', 30)
plt.rcParams.update({
    "text.usetex": True,
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


##################################### Reading 101 images for each case ###################################


start_time = time.time()

base_dir = '/home/pdp19/Dropbox/1. PhD/1. Academic/ML_PROJECT/Simulations/int_osc_clean/RUNS/FINAL_RUNS'
selected_X_values = [251]

dfs = []

for X in selected_X_values:
    dir_name = f'run_osc_clean_{X}'
    data_images_dir = os.path.join(base_dir, dir_name, 'data_images')

    for filename in os.listdir(data_images_dir):
        if filename.endswith('.npy'):
            Y = int(filename.split('_')[-1].split('.')[0])
            filepath = os.path.join(data_images_dir, filename)
            df = pd.DataFrame(np.load(filepath), columns=["time", "Points_0", "Points_1", "Points_2", "H"])
            matrix_df = df.pivot(index='Points_2', columns='Points_0', values='H').iloc[::-1].to_numpy().flatten()
            matrix_df = pd.DataFrame(matrix_df).T
            matrix_df['Run_ID'] = dir_name
            matrix_df['Time'] = Y
            dfs.append(matrix_df)

result_df = pd.concat(dfs, ignore_index=True)
result_df.sort_values(by=['Run_ID', 'Time'], inplace=True)

# Define the CSV file path
output_csv_file = 'data_images_run_osc_clean_251.csv'

# Check if the file exists to decide on appending and header writing
file_exists = os.path.exists(output_csv_file)

# Append or write the DataFrame to the CSV file without repeating headers if appending
result_df.to_csv(output_csv_file, mode='a' if file_exists else 'w', header=not file_exists, index=False)

# Record the end time and calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total time taken to read files: {elapsed_time} seconds")

print(result_df.tail(5))


# start_time = time.time()

# # # Define the base directory where all the 'run_osc_clean_X' directories are located
# base_dir = '/home/pdp19/Dropbox/1. PhD/1. Academic/ML_PROJECT/Simulations/int_osc_clean/RUNS/FINAL_RUNS'

# # Manually specify the values of X you want to process
# selected_X_values = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

# # Initialize an empty list to store dataframes
# dfs = []

# # Loop through each specified value of X
# for X in selected_X_values:
#     dir_name = f'run_osc_clean_{X}'

#     # Define the path to the 'data_images' directory
#     data_images_dir = os.path.join(base_dir, dir_name, 'data_images')

#     # Loop through each CSV file in the 'data_images' directory
#     for filename in os.listdir(data_images_dir):
#         if filename.endswith('.npy'):
#             # Extract the Y value from the filename
#             Y = int(filename.split('_')[-1].split('.')[0])

#             # Read the CSV file into a dataframe
#             filepath = os.path.join(data_images_dir, filename)
#             #df = pd.read_csv(filepath)
#             df = pd.DataFrame(np.load(filepath), columns=["time", "Points_0","Points_1","Points_2","H"])
#             # Process the dataframe as needed
#             matrix_df = df.pivot(index='Points_2', columns='Points_0', values='H')
#             matrix_df = matrix_df.iloc[::-1]
#             matrix_df = matrix_df.to_numpy().flatten()
#             matrix_df = pd.DataFrame(matrix_df).T

#             # Add columns with information
#             matrix_df['Run_ID'] = dir_name
#             matrix_df['Time'] = Y

#             # Append the dataframe to the list
#             dfs.append(matrix_df)

# # Concatenate all dataframes into one
# result_df = pd.concat(dfs, ignore_index=True)

# # Sort the DataFrame if needed
# result_df.sort_values(by=['Run_ID', 'Time'], inplace=True)

# # Save the resulting DataFrame to a CSV file
# result_df.to_csv('data_images_compiled.csv', index=False)
# print(result_df)
# np.save('data_images_compiled', result_df)

# # Record the end time
# end_time = time.time()

# # Calculate and print the total time taken
# elapsed_time = end_time - start_time
# print(f"Total time taken to read files: {elapsed_time} seconds")










# result_df = np.load('data_images_compiled.npy')


# result_df = pd.DataFrame(result_df)


# # Initialize an empty list to store dataframes
# dfs = []

# # Loop through each 'run_osc_clean_X' directory
# for dir_name in os.listdir(base_dir):
#     if dir_name.startswith('run_osc_clean_'):
#         # Get the numerical value X
#         X = dir_name.split('_')[-1]

#         # Define the path to the 'data_images' directory
#         data_images_dir = os.path.join(base_dir, dir_name, 'data_images')

#         # Loop through each CSV file in the 'data_images' directory
#         for filename in os.listdir(data_images_dir):
#             if filename.endswith('.npy'):
#                 # Extract the Y value from the filename
#                 Y = int(filename.split('_')[-1].split('.')[0])

#                 # Read the CSV file into a dataframe
#                 filepath = os.path.join(data_images_dir, filename)
#                 #df = pd.read_csv(filepath)
#                 df = pd.DataFrame(np.load(filepath), columns=["time", "Points_0","Points_1","Points_2","H"])

#                 matrix_df = df.pivot(index='Points_2', columns='Points_0', values='H')
#                 matrix_df = matrix_df.iloc[::-1]
#                 #matrix_df['time'] = filepath.split('_')[-1].split('.')[0]

#                 matrix_df = matrix_df.to_numpy().flatten()
#                 matrix_df = pd.DataFrame(matrix_df).T

#                 # # Add a column with the whole name of the directory
#                 matrix_df['Run_ID'] = dir_name
#                 matrix_df['Time'] = Y

#                 # Append the dataframe to the list
#                 dfs.append(matrix_df)

# # Concatenate all dataframes into one
# result_df = pd.concat(dfs, ignore_index=True)

# # Record the end time
# end_time = time.time()
# # Calculate and print the total time taken
# elapsed_time = end_time - start_time
# print(f"Total time taken to read files: {elapsed_time} seconds")

# np.save('data_images_compiled', result_df)
# result_df.to_csv('data_images_compiled.csv')
