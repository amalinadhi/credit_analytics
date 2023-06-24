import pandas as pd
import utils as utils

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


class ClientData(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30DaysLate: int
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60DaysLate: int
    NumberOfDependents: int

class OutputPrediction(BaseModel):
    label: int
    message: str

def transform_imputer(data, constant_imputer, median_imputer):
    """Function to transform imputer"""
    data = data.copy()

    # Transform
    impute_constant = constant_imputer.transform(data[[CONFIG_DATA['constant_imputer_col']]])
    impute_median = median_imputer.transform(data[[CONFIG_DATA['median_imputer_col']]])

    # Join transformed data
    data[CONFIG_DATA['constant_imputer_col']] = impute_constant
    data[CONFIG_DATA['median_imputer_col']] = impute_median

    return data

def transform_standardize(data, standardizer):
    """Function to standardize data"""
    data_standard = pd.DataFrame(standardizer.transform(data))
    data_standard.columns = data.columns
    data_standard.index = data.index
    return data_standard


# LOAD PREPROCESSOR & MODEL
CONFIG_DATA = utils.config_load()
preprocessor = utils.pickle_load(CONFIG_DATA['preprocessor_path'])
constant_imputer = preprocessor['constant_imputer']
median_imputer = preprocessor['median_imputer']
standardizer = preprocessor['standardizer']
model = utils.pickle_load(CONFIG_DATA['best_model_path'])
threshold = utils.pickle_load(CONFIG_DATA['best_threshold_path'])


app = FastAPI()

@app.get('/')
def home():
    return {'text': 'our first route'}

@app.post('/predict/')
def predict(data: ClientData):
    # Convert data api to dataframe
    client_df = pd.DataFrame(data).set_index(0).T.reset_index(drop=True)
    
    # Correct the column name
    client_df = client_df.rename(columns = {
        'NumberOfTime30DaysLate': 'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTime60DaysLate': 'NumberOfTime60-89DaysPastDueNotWorse'
    })

    # Preprocess Data
    X = client_df
    X_imputed = transform_imputer(data = X,
                                  constant_imputer = constant_imputer,
                                  median_imputer = median_imputer)
    X_clean = transform_standardize(data = X_imputed,
                                    standardizer = standardizer)

    # Predict data
    y_pred_proba = model.predict_proba(X_clean)[:, 1]
    y_pred = int((y_pred_proba >= threshold).astype(int)[0])
    y_label = 'not default'
    if y_pred == 1:
        y_label = 'default'

    return {"res" : y_pred,
            "label": y_label, 
            "error_msg": ""}



if __name__ == '__main__':
    uvicorn.run('api:app',
                host = '127.0.0.1',
                port = 8000)