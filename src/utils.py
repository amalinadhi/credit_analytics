import yaml
import joblib
from datetime import datetime


CONFIG_DIR = 'config/config.yaml'


def config_load():
    """Function to load config files"""
    try:
        with open(CONFIG_DIR, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as error:
        raise RuntimeError('Parameters file not found in path.')
    
    return config

def pickle_load(file_path):
    """Function to load pickle files"""
    return joblib.load(file_path)

def pickle_dump(data, file_path):
    """Function to dump data into pickle"""
    joblib.dump(data, file_path)

def time_stamp():
    """Function that return the current time"""
    return datetime.now()

params = config_load()
PRINT_DEBUG = params['print_debug']
def print_debug(messages):
    """Function to print a debug message in terminal"""
    if PRINT_DEBUG:
        print(time_stamp(), messages)
    