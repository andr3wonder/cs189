from collections import Counter
import numpy as np
from numpy import genfromtxt
import scipy.io
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from pydot import graph_from_dot_data
from scipy import stats
import io
import os
import matplotlib.pyplot as plt
import pandas as pd


import random
np.random.seed(1010)

eps = 1e-5  # a small number

# Part 4.1

class DecisionTree:

    def __init__(self, max_depth=5, feature_labels=None, m=None, random_state=None):
        self.max_depth = max_depth
        self.features = feature_labels 
        self.left, self.right = None, None  # for non-leaf nodes

        # best feature and threshold
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes
        self.m = m

    @staticmethod
    def entropy(y):
        # TODO
        if len(y) == 0:
            return 0
        p = np.where(y < 0.5)[0].size / y.shape[0]
        if p < eps or 1 - p < eps: # prob too small 
            return 0
        entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)
        return entropy

    @staticmethod
    def information_gain(X, y, thresh):
        # TODO

        # get entropy
        par_entropy = DecisionTree.entropy(y)

        y1 = y[np.where(X < thresh)[0]]
        p1 = y1.size / len(y)
        y2 = y[np.where(X >= thresh)[0]]
        p2 = y2.size / len(y)

        child_entropy = p1 * DecisionTree.entropy(y1) + p2 * DecisionTree.entropy(y2)
        return par_entropy - child_entropy

    # @staticmethod
    # def gini_impurity(X, y, thresh):
    #     pass

    # @staticmethod
    # def gini_purification(X, y, thresh):
    #     pass

    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        # TODO
        # Check if maximum depth reached
        if self.max_depth > 0:
            # Initialize a list to store the info gains of all possible splits
            gains = []

            if self.m: 
                # If m, select a random subset of d features
                feature_idx = np.random.choice(np.arange(X.shape[1]), size = self.m, replace=False)
            else: 
                # use all features
                feature_idx = np.arange(X.shape[1])

            # Prepare thresholds for potential splits, avoiding exact min/max values to ensure splits that divide data
            # np.linspace(start, stop, num)
            thresh = np.array(
                [np.linspace(np.min(X[:, i]) + eps, np.max(X[:, i]) - eps, num=10)
                 for i in range(X.shape[1])] # loops over each feature 
            )[feature_idx]

            # Populate gains with info gains according to thresh values 2D array: F * T
            for i in range(X.shape[1]):  # loops over each feature
                gains.append([self.information_gain(X[:, feature_idx[i]], y, t) for t in thresh[i, :]])

            # NaN in gains -> 0, happens when splits don't divide the data
            gains = np.nan_to_num(np.array(gains)) # covert to array to calc 

            # find index of largest gain across all features
            max_gain_id = np.argmax(gains)

            # feature index and thresh index
            self.split_idx, thresh_idx = np.unravel_index(
                max_gain_id, gains.shape
            )  # get index of multi-d array given flat array index

            self.thresh = thresh[self.split_idx, thresh_idx]
            self.split_idx = feature_idx[self.split_idx]

            # Split data according to feature & thresh
            X0, y0, X1, y1 = self.split(X, y, idx=self.split_idx, thresh=self.thresh)
            if X0.size > 0 and X1.size > 0:
                self.left = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features, m=self.m
                )
                self.left.fit(X0, y0)

                self.right = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features, m=self.m
                )
                self.right.fit(X1, y1)
            else: # no spliting -> leaf node 
                self.max_depth = 0
                self.data, self.labels = X, y
                self.pred = calculate_mode(y)
        else:
            self.data, self.labels = X, y
            self.pred = calculate_mode(y)
        return self

    def predict(self, X):
        # TODO
        if self.max_depth == 0: # leaf node, contains self.pred 
            # assign all data that reach this node the prediction
            return self.pred * np.ones(X.shape[0])
        else: 
            X0, idx0, X1, idx1 = self.split_test(X, idx=self.split_idx, thresh=self.thresh)
            yhat = np.zeros(X.shape[0]) 

            # recursively predict, idx0: all the indices that go left
            yhat[idx0] = self.left.predict(X0)
            yhat[idx1] = self.right.predict(X1)

            return yhat

    def __repr__(self):
        if self.max_depth == 0:
            return "%s (%s)" % (self.pred, self.labels.size)
        else:
            return "[%s < %s: %s | %s]" % (self.features[self.split_idx],
                                           self.thresh, self.left.__repr__(),
                                           self.right.__repr__())


class BaggedTrees(BaseEstimator, ClassifierMixin):

    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            DecisionTreeClassifier(random_state=i, **self.params)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        # TODO Train n trees without replacement
        for i in range(self.n):
            idx = np.random.choice(np.arange(X.shape[0]), size=X.shape[0], replace=True)
            X_idx = X[idx]
            y_idx = y[idx]
            self.decision_trees[i].fit(X_idx, y_idx)
        return self

    def predict(self, X):
        # TODO take avg of predictions
        pred = [self.decision_trees[i].predict(X) for i in range(self.n)] 
        return (np.mean(pred, axis=0) >= 0.5).astype(int) # >=0.5 -> 1 


class RandomForest(BaggedTrees):

    def __init__(self, params=None, n=200, m=1):
        if params is None:
            params = {}
        self.m = m
        params['max_features'] = m
        super().__init__(params=params, n=n)


# class BoostedRandomForest(RandomForest):

#     def fit(self, X, y):
#         pass

#     def predict(self, X):
#         pass


def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # Temporarily assign -1 to missing data
    data[data == b''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == b'-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack(
        [np.array(data, dtype=float),
         np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        for i in range(data.shape[1]):  # Loop over each feature
            column_data = data[:, i]  # Select the current column
            valid_data_mask = column_data != "-1"
            valid_data = column_data[valid_data_mask].astype(float)
            mode_value = calculate_mode(valid_data)
            data[~valid_data_mask, i] = mode_value

    return data, onehot_features


def evaluate(clf):
    print("Cross validation", cross_val_score(clf, X, y))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [
            (features[term[0]], term[1]) for term in counter.most_common()
        ]
        print("First splits", first_splits)

def calculate_mode(y):
    # Ensure y is a numpy array
    y = np.array(y)

    # Filter out any invalid data if necessary (e.g., NaNs)
    valid_y = y[~np.isnan(y)]

    # Use Counter to count occurrences of each value
    counter = Counter(valid_y)

    # Find the value(s) with the highest count
    most_common = counter.most_common(1)  # Get the most common value
    mode_value = most_common[0][0]  # Extract the value from the list
    return mode_value


def accuracy(dt, X_train, X_val, y_train, y_val):
    # TODO 
    acc_train = []
    acc_val = []
    acc_train.append((dt.predict(X_train) == y_train).sum() / len(X_train))
    acc_val.append((dt.predict(X_val) == y_val).sum() / len(X_val))
    return acc_train, acc_val


if __name__ == "__main__":
    dataset = "titanic"
    # dataset = "spam"
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    N = 100
    if dataset == "titanic":
        # Load titanic data
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]

        labeled_idx = np.where(y != b'')[0]
        y = np.array(y[labeled_idx], dtype=float).astype(int)
        print("\n\nPart (b): preprocessing the titanic dataset")
        X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, 1:]) + onehot_features

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'datasets/spam_data/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features", features)
    print("Train/test size", X.shape, Z.shape)

    print("\n\nPart 0: constant classifier")
    print("Accuracy", 1 - np.sum(y) / y.size)

    # sklearn decision tree
    print("\n\nsklearn's decision tree")
    clf = DecisionTreeClassifier(random_state=0, **params)
    clf.fit(X, y)
    evaluate(clf)
    out = io.StringIO()
    export_graphviz(
        clf, out_file=out, feature_names=features, class_names=class_names)
    # For OSX, may need the following for dot: brew install gprof2dot
    graph = graph_from_dot_data(out.getvalue())
    graph_from_dot_data(out.getvalue())[0].write_pdf("%s-tree.pdf" % dataset)

    # TODO
    # Shuffle the data
    index = np.random.permutation(len(X))
    X, y = X[index], y[index]
    index = int(len(X) * 0.8)
    X_train, X_val = np.split(X, [index])
    y_train, y_val = np.split(y, [index])

    # Decision Tree
    print("\n\nPart 4.4: Simple Decision Tree Performance")
    dt = DecisionTree(max_depth=5, feature_labels=features)
    dt.fit(X_train, y_train)
    train_acu = (dt.predict(X_train) == y_train).sum() / len(X_train)
    val_acu = (dt.predict(X_val) == y_val).sum() / len(X_val)
    print(dataset, " Decision Tree Training Accuracy: ", train_acu)
    print(dataset, " Decision Tree Validation Accuracy: ", val_acu)

    # Random Forest
    print("\n\nPart 4.4: Random Forest Performance")
    rf = RandomForest(params, n=N, m=int(np.sqrt(X.shape[1])))
    rf.fit(X_train, y_train)
    train_acu = (rf.predict(X_train) == y_train).sum() / len(X_train)
    val_acu = (rf.predict(X_val) == y_val).sum() / len(X_val)
    print(dataset, " Random Forest Training Accuracy: ", train_acu)
    print(dataset, " Random Forest Validation Accuracy: ", val_acu)

    # TODO: implement and evaluate!
    if dataset == "spam":
        print("\n\nPart 4.5, 2: inspect the tree")
        dt = DecisionTree(max_depth=3, feature_labels=features)
        dt.fit(X_train, y_train)
        index = np.random.randint(0, X.shape[0], size=2)
        while y[index[0]] == y[index[1]]:
            index = np.random.randint(0, X.shape[0], size=2)
        print(dt)
        print(list(zip(X[index[0]], features)), y[index[0]])
        print(list(zip(X[index[1]], features)), y[index[1]])

        print("\n\nPart 4.5, 3: depth of decision tree")
        train_accuracy = []
        val_accuracy = []
        for depth in range(1, 40):
            params["max_depth"] = depth
            dt = DecisionTree(max_depth=depth, feature_labels=features)
            dt.fit(X_train, y_train)
            train_acu, val_acu = accuracy(dt, X_train, X_val, y_train, y_val)
            train_accuracy.append(train_acu)
            val_accuracy.append(val_acu)
        plt.plot(train_accuracy, label="train accuracy")
        plt.plot(val_accuracy, label="val accuracy")
        plt.xlabel("depth")
        plt.ylabel("train/validation accuracy")
        plt.legend(loc="upper right")
        plt.show()

    if dataset == "titanic":
        dt = DecisionTree(max_depth=3, feature_labels=features)
        dt.fit(X, y)
        print("Part 4.6 titanic tree: \n")
        print(dt)

    # Kaggle
    def save(test_result, name):
        df = pd.DataFrame({"Category": test_result})
        df.index += 1
        df.to_csv(f"{name}.csv", index_label="Id")

    # Kaggle for spam, titanic
    rf = RandomForest(params, n=N, m=int(np.sqrt(X.shape[1])))
    rf.fit(X, y)
    prediction_rf = rf.predict(Z)

    dt = DecisionTree(max_depth=9, feature_labels=features)
    dt.fit(X, y)
    prediction_dt = dt.predict(Z)

    prediction_rf = rf.predict(Z)
    if dataset == "spam":
        save(prediction_rf, "spam")
    else:
        save(prediction_dt, "titanic")
