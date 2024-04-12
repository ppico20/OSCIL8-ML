import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import sys
from contextlib import redirect_stdout
from sklearn.model_selection import train_test_split

# Define the file name for the log
log_file = 'log_Prepare_data_PCA.log'

# Function to redirect stdout to the log file
def redirect_stdout_to_log(log_file):
    sys.stdout = open(log_file, 'a', buffering=1)

# Redirect stdout to the log file
redirect_stdout_to_log(log_file)

# Function to read and preprocess data in parallel
def read_and_preprocess_images(file_path):

    print("Beginning images reading ...")
    result_df = pd.read_csv(file_path)
    result_df = result_df.replace([np.inf, -np.inf], np.nan)
    rows_to_drop = result_df[result_df.isna().any(axis=1)]

    if not rows_to_drop.empty:
        print("Dropping rows for 'Run_ID' and 'Time':")
        print(rows_to_drop[['Run_ID', 'Time']])

    result_df = result_df.dropna()

    print("Finished reading and processing images ...")
    return result_df

def read_and_preprocess_features(file_path):
    
    print("Beginning reading simulation parameters ...")
    data_params = pd.read_csv(file_path)
    data_params = data_params[['Run_ID','Time','Time_tilde','epsilon','rho_r','mu_r','La_l','Bo_l']]
    print("Finished reading simulation parameters ...")
    
    return data_params

def perform_train_test_split(data):
    
    features = data[['Run_ID','Time','Time_tilde','epsilon','rho_r','mu_r','La_l','Bo_l']]
    target = data.drop(columns=['Time_tilde','epsilon','rho_r','mu_r','La_l','Bo_l'])

    X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(features, target, random_state=0)    
    
    return X_train_original, X_test_original, y_train_original, y_test_original

# Main function
if __name__ == "__main__":
    
    images_path = 'data_images_compiled.csv'
    features_path = "../compiled_data/data_params.csv"

    start_time = time.time()

    with ThreadPoolExecutor() as executor:
        # Read and preprocess data in parallel
        future_data = executor.submit(read_and_preprocess_images, images_path)
        future_second_data = executor.submit(read_and_preprocess_features, features_path)
        
        result_df = future_data.result()
        data_params = future_second_data.result()
    
    # Important: this line matches images data with simulation parameters
    merged_df = pd.merge(result_df, data_params, on=['Run_ID', 'Time'], how='left')
    
    end_time = time.time()  
    elapsed_time = end_time - start_time
    print(f"Data reading, processing, and merging completed in {elapsed_time:.2f} seconds.")

    # Performing Train/Test split
    start_time = time.time()

    X_train_original, X_test_original, y_train_original, y_test_original = perform_train_test_split(merged_df)

    X_train_original.to_csv('X_train_original.csv', index = False)
    X_test_original.to_csv('X_test_original.csv', index = False)
    y_train_original.to_csv('y_train_original.csv', index = False)
    y_test_original.to_csv('y_test_original.csv', index = False)

    print('############ DATA SHAPE (INCLUDING RUN_ID AND TIME COLUMNS) ################')
    print("Shape of X_train:",X_train_original.shape)
    print("Shape of X_test:",X_test_original.shape)
    print("Shape of y_train:",y_train_original.shape)
    print("Shape of y_test:",y_test_original.shape)

    end_time = time.time()  
    elapsed_time = end_time - start_time

    print(f"Train/Test split and data writing completed in {elapsed_time:.2f} seconds.")
