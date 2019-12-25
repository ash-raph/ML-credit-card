from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from pandas import read_csv, concat
import matplotlib.pyplot as plt
from math import trunc, ceil
import numpy as np
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import warnings

warnings.filterwarnings("ignore")


class ML(object):
    def __init__(self, path, flods_number=15):
        self.data_path = path
        self.flods_number = flods_number
        self.results = {}
        # self.training_data=None
        # self.test_data=None
        try:
            self.data = read_csv(path)
            # remove unuseful data
            self.data = self.data.drop(columns=['Amount', 'Time'])
        except Exception:
            print('an exception raised while trying to open file ,'
                  ' can you provide a valid csv file path ')

        self.colors = ['red', 'blue', 'orange', 'yellow', 'green', 'black', 'cyan', 'magenta', 'DarkGreen']

    def scale_data(self):
        for col in self.data.keys():
            if col != 'Class':
                self.data[col] = scale(self.data[col])

    def apply_knn(self, n=5):
        knn_classifier = KNeighborsClassifier(n_neighbors=n)
        res = cross_val_score(knn_classifier, self.data.ix[:, :-1], self.data['Class'], cv=self.flods_number)
        self.results.update({'KNN': res})

    def apply_decision_tree(self, depth=2):
        dt_classifier = DecisionTreeClassifier(max_depth=depth)
        res = cross_val_score(dt_classifier, self.data.ix[:, :-1], self.data['Class'], cv=self.flods_number)
        self.results.update({'Decision Tree': res})

    def base_algo(self, algo):
        flod_size = trunc(len(self.data) / self.flods_number)
        data_keys = list(self.data.keys().difference({'Class'}))
        result = []
        for i in range(self.flods_number):
            start = i * flod_size
            end = ((i + 1) * flod_size) if i != (self.flods_number - 1) else len(self.data)

            test = self.data[start:end]
            training = concat([self.data[:start], self.data[end:]])
            res = algo(training, test, data_keys)

            result.append((np.asarray(test['Class']) == res).sum() / (end - start))

        return result

    def apply_naif_bayes(self):
        def naif_bayes(training, test, data_keys):
            gnb = GaussianNB()
            return gnb.fit(training[data_keys], training['Class']).predict(test[data_keys])

        res = self.base_algo(naif_bayes)

        self.results['Naif Bayes'] = res

    def apply_linear_svm(self):
        def linear_svm(training, test, data_keys):
            # may be we should scale data
            # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
            svc = LinearSVC(max_iter=3000)
            return svc.fit(training[data_keys], training['Class']).predict(test[data_keys])

        res = self.base_algo(linear_svm)

        self.results['Linear SVM'] = res

    def apply_svm(self):
        def svm(training, test, data_keys):
            svc = SVC(gamma='scale')
            return svc.fit(training[data_keys], training['Class']).predict(test[data_keys])

        res = self.base_algo(svm)

        self.results['SVM'] = res

    def apply_logistic_regression(self):
        def logistic_regression(training, test, data_keys):
            lr = LogisticRegression(solver='lbfgs')
            return lr.fit(training[data_keys], training['Class']).predict(test[data_keys])

        res = self.base_algo(logistic_regression)

        self.results['Logistic Regression'] = res

    def apply_random_forest(self):
        def random_forest(training, test, data_keys):
            rf = RandomForestClassifier(n_estimators=50, max_depth=9)
            return rf.fit(training[data_keys], training['Class']).predict(test[data_keys])

        res = self.base_algo(random_forest)

        self.results['Random Forest'] = res

    def apply_lda(self):
        def lda(training, test, data_keys):
            lda = LinearDiscriminantAnalysis()
            return lda.fit(training[data_keys], training['Class']).predict(test[data_keys])

        res = self.base_algo(lda)

        self.results['LDA'] = res

    def apply_qda(self):
        def qda(training, test, data_keys):
            lda = QuadraticDiscriminantAnalysis()
            return lda.fit(training[data_keys], training['Class']).predict(test[data_keys])

        res = self.base_algo(qda)

        self.results['QDA'] = res

    def apply_all_algorithms(self):
        functions = [fct for fct in self.__dir__() if 'apply' in fct and fct != 'apply_all_algorithms']
        for fct in functions:
            call_me = self.__getattribute__(fct)
            call_me()

    def plot_curves(self):
        x = [i for i in range(self.flods_number)]
        min_value = min([min(v) for k, v in self.results.items()])
        with plt.style.context('ggplot'):
            i = 1
            for k, v in self.results.items():
                plt.subplot(ceil(len(self.results) / 4), 4, i)
                i += 1
                plt.ylim(min_value, 1.005)
                plt.plot(x, v, color=self.colors[i-2])
                plt.title(k)
