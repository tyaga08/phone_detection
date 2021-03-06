############################################################
########## Class for the SVM Classifier ####################
############################################################

import numpy as np
import sklearn as sk
from sklearn import svm

class svm_clf:
    def __init__(self, data, label):
        self.clf = svm.SVC(gamma=1, decision_function_shape='ovo', kernel='poly', degree=3,  probability=True)#, class_weight={1:5})
        self.X_train, self.X_test, self.y_train, self.y_test = sk.model_selection.train_test_split(data, label, test_size = 0.15, random_state = 42)

# The function called for training the classifier
    def svm_train(self, flag= False, train_data= None, train_label= None, test_data= None, test_label= None):

        if flag==False:
            train_data = self.X_train
            train_label = self.y_train
  
        if test_data==None:
            test_data = self.X_test
        if test_label==None:
            test_label = self.y_test
        
        self.clf.fit(train_data, train_label)
        self.y_predict = self.clf.predict(test_data)
        self.accuracy = sk.metrics.f1_score(test_label, self.y_predict, 'macro')

    def svm_check(self):
        print("let's see the confidence")
        for i in range(0,len(self.X_validate)):
            print(i)
            print(self.y_predict[i], self.y_test[i])
            print(self.clf.decision_function([self.X_test[i]]))
            print()


    
