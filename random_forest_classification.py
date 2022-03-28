# import tensorflow as tf
# from tensorflow import keras
import pandas as pd
import json

########### Create JSON
# train = pd.read_csv("classification_train.csv")
# print(train)
# df = train.reset_index()
# train_list = []
# for index, row in df.iterrows():
#     dict_ = {}
#     for i in range(0, 90):
#         # print(row["c1"], row["c2"])
#         dict_[f"WifiAccessPoint_{i}"] = row[f"WifiAccessPoint_{i}"]
#     dict_["location_coded"] = row["location_coded"]
#     train_list.append(dict_)


# # json_object = json.dumps(train_list)
# with open("train.json", "w") as f:
#     json.dump(train_list, f)

##############
# from sklearn.ensemble import RandomForestClassifier

# vrf =  RandomForestClassifier(n_estimators= 10)
# classes = []
# for i in range(0, 90):
#         classes =f"WifiAccessPoint_{i}"

# vrf.fit(trainingData, null, "room", function(err, trees) {
#   var pred = rf.predict([formattedLiveData], trees);
#   return classes[pred[0]]; // the room predicted.
# });

##############

import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# from whereami.get_data import get_train_data
# from whereami.utils import get_model_file
def get_model_file(path=""):
    return "model_file.pkl"


def get_train_data(path):
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

    return X, y


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
    labels = list(set(y))
    return X, y, labels


def get_pipeline(clf=RandomForestClassifier(n_estimators=100, class_weight="balanced")):
    return make_pipeline(DictVectorizer(sparse=False), clf)


def train_model(path=None):
    model_file = get_model_file(path)
    X, y = get_train_data(path)
    if len(X) == 0:
        raise ValueError("No wifi access points have been found during training")
    # fantastic: because using "quality" rather than "rssi", we expect values 0-150
    # 0 essentially indicates no connection
    # 150 is something like best possible connection
    # Not observing a wifi will mean a value of 0, which is the perfect default.
    lp = RandomForestClassifier(n_estimators=100, class_weight="balanced")
    lp.fit(X, y)
    with open(model_file, "wb") as f:
        pickle.dump(lp, f)
    return lp


def get_model(path=None):
    model_file = get_model_file(path)
    if not os.path.isfile(model_file):  # pragma: no cover
        msg = "First learn a location, e.g. with `whereami learn -l kitchen`."
        # raise LearnLocation(msg)
    with open(model_file, "rb") as f:
        lp = pickle.load(f)
    return lp


def predict(lp, X_test):
    y_pred = lp.predict(X_test)
    return y_pred


if __name__ == "__main__":
    lp = train_model()
    X_test, y_test, labels = get_test_data()
    y_pred = predict(lp, X_test)
    print(accuracy_score(y_test, y_pred))
    # cm = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # plt.show()
