# Add parent directory to path to allow import of config.py
import sys
sys.path.append('..')
import config as conf
import functions.initialization_functions as init_funcs
import functions.data_manipulation_functions as data_funcs
import functions.parallization_functions as parallel_funcs

# import time
import numpy as np
# import pandas as pd
# import statsmodels.api as sm
# from scipy import stats
# from sklearn.linear_model import Ridge

# print(f'Python: {sys.version}')
# print()
# print(f'Numpy {np.__version__}')
# print(f'Pandas {pd.__version__}')

# Load parsed data
block_size = 5

output_file = f'{conf.DATA_PATH}/parsed_data/structured_bootstrap_blocksize{block_size}.npy'
timepoints = np.load(output_file)

# print(f'Timepoints shape: {timepoints.shape}')
# print()
# print('Column types:')

# for column in timepoints[0,0,0,0:]:
#     print(f'\t{type(column)}')

# print()
# print(f'Example block:\n{timepoints[0,0,0:,]}')

# Set run parameters
num_samples = 18
sample_size = 3
model_orders = [4]
model_types = ['OLS', 'TS', 'Seigel', 'Ridge']
time_fit = False

# Fire up the pool
pool, result_objects = parallel_funcs.start_multiprocessing_pool()

# Loop on samples, assigning each to a different worker
for sample_num in range(num_samples):

    result = pool.apply_async(parallel_funcs.parallel_bootstrapped_smape,
        args = (
            timepoints, 
            sample_num, 
            sample_size, 
            model_orders, 
            model_types,
            time_fit
        )
    )

    # Add result to collection
    result_objects.append(result)

# Get and parse result objects, clean up pool
data = parallel_funcs.cleanup_bootstrapping_multiprocessing_pool(pool, result_objects)

for key, value in data.items():
    print(f'{key}: {value}')