import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn import preprocessing


def get_train_data(path=None):
    train = pd.read_csv("classification_train.csv").fillna(0)
    # train = df1.replace(to_replace=np.nan, value=None, inplace=True)
    print(train)
    df = train.reset_index()
    X = []
    y = []
    for index, row in df.iterrows():
        x = []
        for i in range(0, 90):
            x.append(row[f"WifiAccessPoint_{i}"])
        #     # print(row["c1"], row["c2"])
        #     dict_[f"WifiAccessPoint_{i}"] = row[f"WifiAccessPoint_{i}"]
        # train_list.append(dict_)
        X.append(x)
        y.append(row["location_coded"])
    labels = list(set(y))
    return X, y, labels


def get_test_data(path=None):
    test = pd.read_csv("classification_test.csv").fillna(0)
    # train = df1.replace(to_replace=np.nan, value=None, inplace=True)
    print(test)
    df = test.reset_index()
    X = []
    y = []
    for index, row in df.iterrows():
        x = []
        for i in range(0, 90):
            x.append(row[f"WifiAccessPoint_{i}"])
        #     # print(row["c1"], row["c2"])
        #     dict_[f"WifiAccessPoint_{i}"] = row[f"WifiAccessPoint_{i}"]
        # train_list.append(dict_)
        X.append(x)
        y.append(row["location_coded"])
    return X, y


def preprocess(x, y):
    #   x = tf.cast(x, tf.float32) / 255.0
    #   y = tf.cast(y, tf.int64)

    return x, y


def create_dataset(xs, ys, n_classes=10):
    ys = tf.one_hot(ys, depth=n_classes)
    return (
        tf.data.Dataset.from_tensor_slices((xs, ys))
        .map(preprocess)
        .shuffle(len(ys))
        .batch(128)
    )


le = preprocessing.LabelEncoder()
import numpy as np

if __name__ == "__main__":
    x_train, y_train, labels = get_train_data()
    x_val, y_val = get_test_data()
    n_classes = len(labels)

    # Changing labels to numbers
    le.fit(labels)
    y_train = le.transform(y_train).tolist()
    y_val = le.transform(y_val).tolist()

    print("len(x_val)", len(x_val))
    # print(x_val)
    print("len(y_val)", len(y_val))
    train_dataset = create_dataset(x_train, y_train, n_classes=n_classes)
    val_dataset = create_dataset(x_val, y_val, n_classes=n_classes)

    model = keras.Sequential(
        [
            keras.layers.Dense(units=128, activation="relu"),
            keras.layers.Dense(units=256, activation="relu"),
            keras.layers.Dense(units=512, activation="relu"),
            keras.layers.Dense(units=1024, activation="relu"),
            keras.layers.Dense(units=512, activation="relu"),
            keras.layers.Dense(units=256, activation="relu"),
            keras.layers.Dense(units=128, activation="relu"),
            keras.layers.Dense(units=n_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(
        train_dataset.repeat(),
        epochs=40,
        steps_per_epoch=500,
        validation_data=val_dataset.repeat(),
        validation_steps=2,
    )

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    y_val = tf.one_hot(y_val, depth=n_classes)
    results = model.evaluate(np.asarray(x_val), y_val, batch_size=128)
    print("test loss, test acc:", results)
