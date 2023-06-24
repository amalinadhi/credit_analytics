import pandas as pd
import utils as utils

from sklearn.model_selection import train_test_split


def read_data(return_file=False):
    # Read data
    data = pd.read_csv(CONFIG_DATA['raw_dataset_path'], 
                       sep=',',
                       index_col=CONFIG_DATA['index_column'])

    # Print data
    print('data shape   :', data.shape)

    # Dump data
    utils.pickle_dump(data, CONFIG_DATA['data_set_path'])

    # Return data
    if return_file:
        return data

def split_input_output(return_file=False):
    # Read data
    data = utils.pickle_load(CONFIG_DATA['data_set_path'])

    # Split input & output
    y = data[CONFIG_DATA['output_column']]
    X = data.drop([CONFIG_DATA['output_column']], axis=1)

    # Print splitting
    print('Input shape  :', X.shape)
    print('Output shape :', y.shape)
    print('Input NAN    :')
    print(X.isnull().sum())
    print('Benchmark    :')
    print(y.value_counts(normalize=True))
    
    # Dump file
    utils.pickle_dump(X, CONFIG_DATA['input_set_path'])
    utils.pickle_dump(y, CONFIG_DATA['output_set_path'])
    utils.pickle_dump(X.columns, CONFIG_DATA['input_columns_path'])     # dump input columns

    if return_file:
        return X, y
        
def split_train_test(return_file=True):
    # Load data
    X = utils.pickle_load(CONFIG_DATA['input_set_path'])
    y = utils.pickle_load(CONFIG_DATA['output_set_path'])

    # Split test & rest (train & valid)
    X_train, X_test, y_train, y_test = train_test_split(
                                            X,
                                            y,
                                            test_size = CONFIG_DATA['test_size'],
                                            random_state = CONFIG_DATA['seed']
                                        )
    
    # Split train & valid
    X_train, X_valid, y_train, y_valid = train_test_split(
                                            X_train,
                                            y_train,
                                            test_size = CONFIG_DATA['test_size'],
                                            random_state = CONFIG_DATA['seed']
                                        )
    
    # Print splitting
    print('X_train shape :', X_train.shape)
    print('y_train shape :', y_train.shape)
    print('X_valid shape  :', X_valid.shape)
    print('y_valid shape  :', y_valid.shape)
    print('X_test shape  :', X_test.shape)
    print('y_test shape  :', y_test.shape)

    # Dump file
    utils.pickle_dump(X_train, CONFIG_DATA['train_set_path'][0])
    utils.pickle_dump(y_train, CONFIG_DATA['train_set_path'][1])
    utils.pickle_dump(X_valid, CONFIG_DATA['valid_set_path'][0])
    utils.pickle_dump(y_valid, CONFIG_DATA['valid_set_path'][1])
    utils.pickle_dump(X_test, CONFIG_DATA['test_set_path'][0])
    utils.pickle_dump(y_test, CONFIG_DATA['test_set_path'][1])

    if return_file:
        return X_train, X_valid, X_test, y_train, y_valid, y_test


if __name__ == '__main__':
    # 1. Load configuration file
    CONFIG_DATA = utils.config_load()

    # 2. Read all raw dataset
    read_data()
    split_input_output()
    split_train_test()