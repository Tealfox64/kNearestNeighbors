from collections import Counter

import numpy as np


class kNNModel(object):
    # this appears to do nothing
    def __init__(self, data_train):
        self.data_train = data_train

    def predict(self, data_train, targets_train, data_test, k):
        targets = []
        distances = []

        # find all the distances
        for d in range(len(data_train)):
            distance = np.sqrt(np.sum(np.square(data_test - data_train[d, :])))
            distances.append([distance, d])

        # sort distances in ascending order
        distances = sorted(distances)

        # make a list of the targets of the nearest k neighbors
        for neighbor in range(k):
            i = distances[neighbor][1]
            targets.append(targets_train[i])

        # return the target that shows up the most
        # use the Counter subclass of a dictionary from the collections library
        # use Counter's most_common function to determine most common results and put them in order
        # return the first data from the first row of it
        return Counter(targets).most_common(1)[0][0]


class kNNClassifier(object):
    def __init__(self, neighbors):
        self.k = neighbors
        self.kNNModel = kNNModel(neighbors)

    # this appears to do nothing
    def fit(self, data_train, targets_train):
        return kNNModel(data_train)
