import pandas as pd
import numpy as np
import copy
import utils as utils

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, f1_score


def create_model_param():
    """Create the model objects"""
    knn_params = {
        'n_neighbors': [50, 100, 200],
    }
    
    lgr_params = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1],
        'max_iter': [100, 300, 500]
    }

    xgb_params = {
        'n_estimators': [5, 10, 25, 50]
    }

    # Create model params
    list_of_param = {
        'KNeighborsClassifier': knn_params,
        'LogisticRegression': lgr_params,
        'XGBClassifier': xgb_params
    }

    return list_of_param

def create_model_object():
    """Create the model objects"""
    print("Creating model objects")

    # Create model objects
    knn = KNeighborsClassifier()
    lgr = LogisticRegression(solver='liblinear')
    xgb = XGBClassifier()

    # Create list of model
    list_of_model = [
        {'model_name': knn.__class__.__name__, 'model_object': knn},
        {'model_name': lgr.__class__.__name__, 'model_object': lgr},
        {'model_name': xgb.__class__.__name__, 'model_object': xgb}
    ]

    return list_of_model

def train_model(return_file=True):
    """Function to get the best model"""
    # Load dataset
    X_train = utils.pickle_load(CONFIG_DATA['train_clean_path'][0])
    y_train = utils.pickle_load(CONFIG_DATA['train_clean_path'][1])
    X_valid = utils.pickle_load(CONFIG_DATA['valid_clean_path'][0])
    y_valid = utils.pickle_load(CONFIG_DATA['valid_clean_path'][1])

    # Create list of params & models
    list_of_param = create_model_param()
    list_of_model = create_model_object()

    # List of trained model
    list_of_tuned_model = {}

    # Train model
    for base_model in list_of_model:
        # Current condition
        model_name = base_model['model_name']
        model_obj = copy.deepcopy(base_model['model_object'])
        model_param = list_of_param[model_name]

        # Debug message
        print('Training model :', model_name)

        # Create model object
        model = RandomizedSearchCV(estimator = model_obj,
                                   param_distributions = model_param,
                                   n_iter=5,
                                   cv = 5,
                                   random_state = 123,
                                   n_jobs=1,
                                   verbose=10,
                                   scoring = 'roc_auc')
        
        # Train model
        model.fit(X_train, y_train)

        # Predict
        y_pred_proba_train = model.predict_proba(X_train)[:, 1]
        y_pred_proba_valid = model.predict_proba(X_valid)[:, 1]
        
        # Get score
        train_score = roc_auc_score(y_train, y_pred_proba_train)
        valid_score = roc_auc_score(y_valid, y_pred_proba_valid)

        # Append
        list_of_tuned_model[model_name] = {
            'model': model,
            'train_auc': train_score,
            'valid_auc': valid_score,
            'best_params': model.best_params_
        }

        print("Done training")
        print("")

    # Dump data
    utils.pickle_dump(list_of_param, CONFIG_DATA['list_of_param_path'])
    utils.pickle_dump(list_of_model, CONFIG_DATA['list_of_model_path'])
    utils.pickle_dump(list_of_tuned_model, CONFIG_DATA['list_of_tuned_model_path'])

    if return_file:
        return list_of_param, list_of_model, list_of_tuned_model    

def get_best_model(return_file=True):
    """Function to get the best model"""
    # Load tuned model
    list_of_tuned_model = utils.pickle_load(CONFIG_DATA['list_of_tuned_model_path'])

    # Get the best model
    best_model_name = None
    best_model = None
    best_performance = -99999
    best_model_param = None

    for model_name, model in list_of_tuned_model.items():
        if model['valid_auc'] > best_performance:
            best_model_name = model_name
            best_model = model['model']
            best_performance = model['valid_auc']
            best_model_param = model['best_params']

    # Dump the best model
    utils.pickle_dump(best_model, CONFIG_DATA['best_model_path'])

    # Print
    print('=============================================')
    print('Best model        :', best_model_name)
    print('Metric score      :', best_performance)
    print('Best model params :', best_model_param)
    print('=============================================')

    if return_file:
        return best_model

def get_best_threshold(return_file=True):
    """Function to tune & get the best decision threshold"""
    # Load data & model
    X_valid = utils.pickle_load(CONFIG_DATA['valid_clean_path'][0])
    y_valid = utils.pickle_load(CONFIG_DATA['valid_clean_path'][1])
    best_model = utils.pickle_load(CONFIG_DATA['best_model_path'])

    # Get the proba pred
    y_pred_proba = best_model.predict_proba(X_valid)[:, 1]

    # Initialize
    metric_threshold = pd.Series([])
    
    # Optimize
    for threshold_value in THRESHOLD:
        # Get predictions
        y_pred = (y_pred_proba >= threshold_value).astype(int)

        # Get the F1 score
        metric_score = f1_score(y_valid, y_pred, average='macro')

        # Add to the storage
        metric_threshold[metric_score] = threshold_value

    # Find the threshold @max metric score
    metric_score_max_index = metric_threshold.index.max()
    best_threshold = metric_threshold[metric_score_max_index]
    print('=============================================')
    print('Best threshold :', best_threshold)
    print('Metric score   :', metric_score_max_index)
    print('=============================================')
    
    # Dump file
    utils.pickle_dump(best_threshold, CONFIG_DATA['best_threshold_path'])

    if return_file:
        return best_threshold


if __name__ == '__main__':
    # 1. Load configuration file & Set Threshold
    CONFIG_DATA = utils.config_load()
    THRESHOLD = np.linspace(0, 1, 100)

    # 2. Train & Optimize the model
    train_model()

    # 3. Get the best model
    get_best_model()

    # 4. Get the best threshold for the best model
    get_best_threshold()