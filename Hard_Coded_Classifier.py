class HardCodedModel(object):
    def predict(self, data_test):
        targets = []
        for d in data_test:
            targets.append(0)
        return targets


class HardCodedClassifier(object):
    def __init__(self):
        self.HardCodedModel = HardCodedModel()

    def fit(self, data_train, targets_train):
        return HardCodedModel()
