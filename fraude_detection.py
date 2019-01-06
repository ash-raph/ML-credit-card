from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from pandas import read_csv, concat
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from math import trunc
import numpy as np


class ML(object):
    def __init__(self, path, flods_number=15):
        self.data_path = path
        self.flods_number = flods_number
        self.results = {}
        # self.training_data=None
        # self.test_data=None
        try:
            self.data = read_csv(path)
        except Exception:
            print('an exception raised while trying to open file ,'
                  ' can you provide a valid csv file path ')
    #
    # def separate_training_and_test_data(self):
    #     # test data will take 25% from all data
    #     classes = list(set(self.data['Class']))
    #     class1 = self.data[self.data == classes[0]]
    #     class2 = self.data[self.data == classes[1]]
    #     test_lenght_class1 = trunc(len(class1) * 0.25)
    #     test_lenght_class2 = trunc(len(class2) * 0.25)
    #     self.test_data = shuffle(concat([class1[:test_lenght_class1], class2[:test_lenght_class2]]))
    #     self.training_data = shuffle(concat([class1[test_lenght_class1:], class2[test_lenght_class2:]]))

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    def apply_knn(self, n=5):
        knn_classifier = KNeighborsClassifier(n_neighbors=n)
        res = cross_val_score(knn_classifier, self.data.ix[:, :-1], self.data['Class'], cv=self.flods_number)
        self.results.update({'KNN': res})

    def apply_decision_tree(self, depth=15):
        dt_classifier = DecisionTreeClassifier(max_depth=depth)
        res = cross_val_score(dt_classifier, self.data.ix[:, :-1], self.data['Class'], cv=self.flods_number)
        self.results.update({'Decision Tree': res})

    def apply_naif_bayes(self):
        flod_size = trunc(len(self.data) / self.flods_number)
        data_keys = list(self.data.keys().difference({'Class'}))
        result = []
        for i in range(self.flods_number):
            start = i * flod_size
            end = ((i + 1) * flod_size) if i != (self.flods_number - 1) else len(self.data)

            test = self.data[start:end]
            training = concat([self.data[:start], self.data[end:]])

            gnb = GaussianNB()
            res = gnb.fit(training[data_keys], training['Class']).predict(test[data_keys])

            result.append((np.asarray(test['Class']) == res).sum() / (end - start))

        self.results['naif bayes'] = result

    def apply_linear_svm(self):
        flod_size = trunc(len(self.data) / self.flods_number)
        data_keys = list(self.data.keys().difference({'Class'}))
        result = []
        for i in range(self.flods_number):
            start = i * flod_size
            end = ((i + 1) * flod_size) if i != (self.flods_number - 1) else len(self.data)

            test = self.data[start:end]
            training = concat([self.data[:start], self.data[end:]])

            svc = LinearSVC()
            res = svc.fit(training[data_keys], training['Class']).predict(test[data_keys])

            result.append((np.asarray(test['Class']) == res).sum() / (end - start))

        self.results['linear svm'] = result

    def apply_svm(self):
        flod_size = trunc(len(self.data) / self.flods_number)
        data_keys = list(self.data.keys().difference({'Class'}))
        result = []
        for i in range(self.flods_number):
            start = i * flod_size
            end = ((i + 1) * flod_size) if i != (self.flods_number - 1) else len(self.data)

            test = self.data[start:end]
            training = concat([self.data[:start], self.data[end:]])

            svc = SVC()
            res = svc.fit(training[data_keys], training['Class']).predict(test[data_keys])

            result.append((np.asarray(test['Class']) == res).sum() / (end - start))

        self.results['svm'] = result

    def plot_curves(self):
        x = [i for i in range(self.flods_number)]
        with plt.style.context('ggplot'):
            for k, v in self.results.items():
                plt.plot(x, v)


