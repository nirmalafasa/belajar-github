import os
import pickle
import sys
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from exception import CustomException
from logger import logging


def save_object(file_path: str, obj: object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as f:
            pickle.dump(obj, f)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str) -> object:
    try:
        with open(file_path, "rb") as obj:
            logging.info("Success to load processor or model")
            return pickle.load(obj)
    except Exception as e:
        logging.info("Fail to load processor or model")
        raise CustomException(e, sys)


def evaluation(
    models: Dict[str, object],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    results = []
    for ml_name, ml_func in models.items():
        ml_func.fit(X_train, y_train)

        pred_train = ml_func.predict(X_train)
        pred_test = ml_func.predict(X_test)

        r2_score_train = r2_score(y_train, pred_train)
        r2_score_test = r2_score(y_test, pred_test)

        rmse_train = mean_squared_error(
            np.exp(y_train), np.exp(pred_train), squared=False
        )
        rmse_test = mean_squared_error(np.exp(y_test), np.exp(pred_test), squared=False)

        results.append(
            {
                "Model Name": ml_name,
                "Model Function": ml_func,
                "R2 Score Train": r2_score_train,
                "R2 Score Test": r2_score_test,
                "RMSE Train": rmse_train,
                "RMSE Test": rmse_test,
            }
        )
    df_results = (
        pd.DataFrame(results)
        .sort_values(["R2 Score Test", "RMSE Test"], ascending=[False, True])
        .reset_index(drop=True)
    )
    df_results["Rank"] = df_results.index + 1
    return df_results
