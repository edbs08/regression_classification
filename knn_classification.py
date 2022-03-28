import pandas as pd
from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import svm


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


def predict(lp, X_test):
    y_pred = lp.predict(X_test)
    return y_pred


if __name__ == "__main__":
    le = preprocessing.LabelEncoder()
    x_train, y_train, labels = get_train_data()
    X_test, y_test = get_test_data()
    n_classes = len(labels)

    # Changing labels to numbers
    le.fit(labels)
    y_train = le.transform(y_train).tolist()
    y_test = le.transform(y_test).tolist()

    # model = KNeighborsClassifier(n_neighbors=3)
    model = svm.SVR()
    model.fit(x_train, y_train)
    y_pred = predict(model, X_test)
    print(accuracy_score(y_test, y_pred))
    print(list(zip(y_test, y_pred)))
