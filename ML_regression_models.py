import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import NearestCentroid
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
import csv

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)
import matplotlib.pyplot as plt


def get_train_data(path=None, column="x"):
    train = pd.read_csv("regression_train.csv").fillna(0)
    df = train.reset_index()
    X = []
    y = []
    for index, row in df.iterrows():
        x = []
        for i in range(0, 90):
            x.append(row[f"WifiAccessPoint_{i}"])
        X.append(x)
        y.append(row[column])

    return X, y


def get_test_data(path=None, column="x"):
    test = pd.read_csv("regression_test.csv").fillna(0.0)
    df = test.reset_index()
    X = []
    y = []
    for index, row in df.iterrows():
        x = []
        for i in range(0, 90):
            x.append(row[f"WifiAccessPoint_{i}"])
        X.append(x)
        y.append(row[column])
    return (
        X,
        y,
    )


from sklearn.neighbors import KNeighborsRegressor


def train_model(reg_column, path=None):
    X, y = get_train_data(column=reg_column)
    # Uncomment the regression model to use
    # lp = RandomForestRegressor(n_estimators=300)
    # lp = svm.SVR()
    lp = KNeighborsRegressor(n_neighbors=2)
    lp.fit(X, y)
    return lp


def predict(lp, X_test):
    y_pred = lp.predict(X_test)
    return y_pred


if __name__ == "__main__":
    reg_column = "x"
    fields = ["x_pred", "x_acual"]
    lp = train_model(reg_column)
    X_test, y_test = get_test_data(column=reg_column)
    y_pred = predict(lp, X_test)
    print(f" MSE for column {reg_column}: ", mean_squared_error(y_test, y_pred))
    print(f" r2_score for column {reg_column}: ", r2_score(y_test, y_pred))
    with open("x_prediction.csv", "w", newline="") as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(list(zip(y_test, y_pred)))

    reg_column = "y"
    lp = train_model(reg_column)
    X_test, y_test = get_test_data(column=reg_column)
    y_pred = predict(lp, X_test)
    print(f" MSE for column {reg_column}: ", mean_squared_error(y_test, y_pred))
    print(f" r2_score for column {reg_column}: ", r2_score(y_test, y_pred))
    fields = ["y_pred", "y_acual"]
    with open("y_prediction.csv", "w", newline="") as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(list(zip(y_test, y_pred)))
