from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def preparar_dados_regressao(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    cols_x = [
        "discounted_price_num",
        "actual_price_num",
        "discount_percentage_num",
        "rating_count_num",
    ]
    dados = df[cols_x + ["rating_num"]].dropna()
    X = dados[cols_x].values
    y = dados["rating_num"].values
    return X, y


def treinar_modelo_rating(df: pd.DataFrame):
    X, y = preparar_dados_regressao(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    return model, {"r2": r2, "rmse": rmse}


def prever_rating(model: LinearRegression, features: list) -> float:
    arr = np.array(features, dtype=float).reshape(1, -1)
    return float(model.predict(arr)[0])
