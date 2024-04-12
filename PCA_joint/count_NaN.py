import pandas as pd
import numpy as np

# Function to find the indexes of NaN or Inf values
def find_nan_inf_indexes(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Finding NaN or Inf values
    # Using np.inf to detect both positive and negative infinity
    is_inf = df.isin([np.inf, -np.inf])
    is_nan = df.isna()

    # Combining the conditions for NaN or Inf
    condition = is_inf | is_nan

    # Iterate over the DataFrame to find the row and column indexes
    for col in condition.columns:
        rows_with_issues = condition.index[condition[col]].tolist()
        if rows_with_issues:  # If the list is not empty
            print(f"Column '{col}' contains NaN/Inf at rows: {rows_with_issues}")

# Example usage
csv_file_path = 'data_images_compiled_1_69.csv'
find_nan_inf_indexes(csv_file_path)
