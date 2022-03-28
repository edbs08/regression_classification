import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn import preprocessing
import numpy as np
import tensorflow_addons as tfa
import csv
import numpy as np


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


def preprocess(x, y):
    return x, y


def create_dataset(xs, ys):
    return np.asarray(xs), np.asarray(ys)


if __name__ == "__main__":
    le = preprocessing.LabelEncoder()
    x_train, y_train = get_train_data()
    x_val, y_val = get_test_data()

    train_dataset = create_dataset(x_train, y_train)
    val_dataset = create_dataset(x_val, y_val)

    model = keras.Sequential(
        [
            keras.layers.Dense(units=128, activation="relu"),
            keras.layers.Dense(units=256, activation="relu"),
            keras.layers.Dense(units=256, activation="relu"),
            keras.layers.Dense(units=128, activation="relu"),
            keras.layers.Dense(units=1, activation="linear"),
        ]
    )

    model.compile(
        loss="mean_squared_error",
        optimizer="adam",
        metrics=["mean_squared_error"],
    )

    history = model.fit(
        np.asarray(x_train),
        np.asarray(y_train),
        epochs=200,
        steps_per_epoch=500,
    )

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(np.asarray(x_val), np.asarray(y_val), batch_size=56)
    y_pred = model.predict(np.asarray(x_val))
    metric = tfa.metrics.r_square.RSquare()
    y_pred = [item for sublist in y_pred for item in sublist]
    metric.update_state(np.asarray(y_val), y_pred)
    r2_result = metric.result()

    print("mean_squared_error:", results)
    print("r_2:", r2_result)

    fields = ["x_pred", "x_acual"]
    with open("x_prediction.csv", "w", newline="") as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(list(zip(y_val, y_pred)))
