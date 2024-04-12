import os
import numpy as np
import pandas as pd

df = pd.read_csv('data_images_compiled_ALL.csv')

# Convert the dataframe to a NumPy array
np_array = df.to_numpy()

# Save the NumPy array as an npy file
np.save('data_images_compiled_ALL.npy', np_array)