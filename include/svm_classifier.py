import numpy as np
import sklearn as sk
from sklearn import svm
# from sklearn.model_selection import train_test_split

class svm_clf:
    def __init__(self, data, label):
        self.clf = svm.SVC(gamma = 'auto')
        self.X_train, self.X_test, self.y_train, self.y_test = sk.model_selection.train_test_split(data, label, test_size = 0.2, random_state = 42)
    
    def svm_train(self):
        self.clf.fit(self.X_train, self.y_train)
        self.y_predict = self.clf.predict(self.X_test)
        print(self.y_predict)
        print(self.y_test)
        self.accuracy = sk.metrics.f1_score(self.y_test, self.y_predict, 'micro')
        print(self.accuracy)
    
