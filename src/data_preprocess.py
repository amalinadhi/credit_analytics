import pandas as pd
import numpy as np
import utils as utils

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from imblearn.under_sampling import RandomUnderSampler


def load_dataset(return_file=True):
    # Load train data
    X_train = utils.pickle_load(CONFIG_DATA['train_set_path'][0])
    y_train = utils.pickle_load(CONFIG_DATA['train_set_path'][1])

    # Load valid data
    X_valid = utils.pickle_load(CONFIG_DATA['valid_set_path'][0])
    y_valid = utils.pickle_load(CONFIG_DATA['valid_set_path'][1])

    # Load test data
    X_test = utils.pickle_load(CONFIG_DATA['test_set_path'][0])
    y_test = utils.pickle_load(CONFIG_DATA['test_set_path'][1])

    # Print 
    print("X_train shape :", X_train.shape)
    print("y_train shape :", y_train.shape)
    print("X_valid shape :", X_valid.shape)
    print("y_valid shape :", y_valid.shape)
    print("X_test shape  :", X_test.shape)
    print("y_test shape  :", y_test.shape)

    if return_file:
        return X_train, X_valid, X_test, y_train, y_valid, y_test
    
def clean_late_data(X, y):
    """Function to clean NumberOfTimes90DaysLate columns"""
    # Find index to drop
    drop_condition = X[CONFIG_DATA['clean_late_col']] >= CONFIG_DATA['clean_late_val']
    index_to_drop = X[drop_condition].index.tolist()

    # Drop data
    X_drop = X.drop(index = index_to_drop)
    y_drop = y.drop(index = index_to_drop)

    # Print
    print("X shape :", X_drop.shape)
    print("y shape :", y_drop.shape)

    return X_drop, y_drop

def clean_unsecured_data(X, y):
    """Function to clean RevolvingUtilizationOfUnsecuredLines columns from outlier"""
    # Find upper boundary & lower boundary
    q1, q3 = np.quantile(X[CONFIG_DATA['clean_unsecure_col']], q = [0.25, 0.75])
    iqr = q3-q1
    ub = q3 + 1.5*iqr
    lb = q1 - 1.5*iqr

    # Filter data
    drop_condition_1 = X[CONFIG_DATA['clean_unsecure_col']] > ub
    drop_condition_2 = X[CONFIG_DATA['clean_unsecure_col']] < lb
    index_to_drop = X[drop_condition_1 | drop_condition_2].index.tolist()

    # Drop data
    X_drop = X.drop(index = index_to_drop)
    y_drop = y.drop(index = index_to_drop)

    # Print
    print("X shape :", X_drop.shape)
    print("y shape :", y_drop.shape)

    return X_drop, y_drop

def fit_imputer(data, return_file=True):
    """Function to fit imputer (constant & median)"""
    # Create imputer
    constant_imputer = SimpleImputer(missing_values = np.nan,
                                     strategy = "constant",
                                     fill_value = CONFIG_DATA['constant_imputer_val'])
    
    median_imputer = SimpleImputer(missing_values = np.nan,
                                   strategy = "median")
    
    # Fit imputer
    constant_imputer.fit(data[[CONFIG_DATA['constant_imputer_col']]])
    median_imputer.fit(data[[CONFIG_DATA['median_imputer_col']]])

    # Dump imputer
    utils.pickle_dump(constant_imputer, CONFIG_DATA['constant_imputer_path'])
    utils.pickle_dump(median_imputer, CONFIG_DATA['median_imputer_path'])

    if return_file:
        return constant_imputer, median_imputer
    
def transform_imputer(data, constant_imputer, median_imputer):
    """Function to transform imputer"""
    data = data.copy()

    # Transform
    impute_constant = constant_imputer.transform(data[[CONFIG_DATA['constant_imputer_col']]])
    impute_median = median_imputer.transform(data[[CONFIG_DATA['median_imputer_col']]])

    # Join transformed data
    data[CONFIG_DATA['constant_imputer_col']] = impute_constant
    data[CONFIG_DATA['median_imputer_col']] = impute_median
    
    # print
    print('data shape :', data.shape)

    return data

def fit_standardize(data, return_file=True):
    """Find standardizer data"""
    standardizer = StandardScaler()

    # Fit standardizer
    standardizer.fit(data)

    # Dump standardizer
    utils.pickle_dump(standardizer, CONFIG_DATA['standardizer_path'])
    
    if return_file:
        return standardizer

def transform_standardize(data, standardizer):
    """Function to standardize data"""
    data_standard = pd.DataFrame(standardizer.transform(data))
    data_standard.columns = data.columns
    data_standard.index = data.index
    return data_standard

def random_undersampler(X, y):
    """Function to under sample the majority data"""
    # Create resampling object
    ros = RandomUnderSampler(random_state = CONFIG_DATA['seed'])

    # Balancing the set data
    X_resample, y_resample = ros.fit_resample(X, y)

    # Print
    print('Distribution before resampling :')
    print(y.value_counts())
    print("")
    print('Distribution after resampling  :')
    print(y_resample.value_counts())

    return X_resample, y_resample

def clean_data(data, constant_imputer, median_imputer, standardizer):
    """Function to clean data"""
    # Impute missing value
    data_imputed = transform_imputer(data, constant_imputer, median_imputer)

    # Standardize data
    data_standard = transform_standardize(data_imputed, standardizer)

    return data_standard

def _preprocess_data(data):
    """Function to preprocess data"""
    # Load preprocessor
    preprocessor = utils.pickle_load(CONFIG_DATA['preprocessor_path'])
    constant_imputer = preprocessor['constant_imputer']
    median_imputer = preprocessor['median_imputer']
    standardizer = preprocessor['standardizer']

    data_clean = clean_data(data,
                            constant_imputer,
                            median_imputer,
                            standardizer)
    
    return data_clean

def generate_preprocessor(return_file=False):
    """Function to generate preprocessor"""
    # Load data
    X = utils.pickle_load(CONFIG_DATA['train_set_path'][0])
    y = utils.pickle_load(CONFIG_DATA['train_set_path'][1])

    # Drop unusual data
    X, y = clean_late_data(X, y)
    X, y = clean_unsecured_data(X, y)

    # Generate preprocessor: imputer
    constant_imputer, median_imputer = fit_imputer(data = X)
    X_imputed = transform_imputer(X, constant_imputer, median_imputer)

    # Generate preprocessor: standardizer
    standardizer = fit_standardize(X_imputed)

    # Dump file
    preprocessor = {
        'constant_imputer': constant_imputer,
        'median_imputer': median_imputer,
        'standardizer': standardizer
    }
    utils.pickle_dump(preprocessor, CONFIG_DATA['preprocessor_path'])
    
    if return_file:
        return preprocessor
    
def preprocess_data(type, return_file=False):
    """Function to preprocess train data"""
    # Load data
    X = utils.pickle_load(CONFIG_DATA[f'{type}_set_path'][0])
    y = utils.pickle_load(CONFIG_DATA[f'{type}_set_path'][1])

    if type == 'train':
        # Drop unusual data
        X, y = clean_late_data(X, y)
        X, y = clean_unsecured_data(X, y)
        
    # Preprocess data
    X_clean = _preprocess_data(X)
    y_clean = y

    # FOR TRAINING ONLY -> DO UNDERSAMPLING
    if type == 'train':
        X_clean, y_clean = random_undersampler(X_clean, y_clean)

    # Print shape
    print("X clean shape:", X_clean.shape)
    print("y clean shape:", y_clean.shape)

    # Dump file
    utils.pickle_dump(X_clean, CONFIG_DATA[f'{type}_clean_path'][0])
    utils.pickle_dump(y_clean, CONFIG_DATA[f'{type}_clean_path'][1])

    if return_file:
        return X_clean, y_clean    


if __name__ == '__main__':
    # 1. Load configuration file
    CONFIG_DATA = utils.config_load()

    # 2. Generate preprocessor
    generate_preprocessor()

    # 3. Preprocess Data
    preprocess_data(type='train')
    preprocess_data(type='valid')
    preprocess_data(type='test')
    