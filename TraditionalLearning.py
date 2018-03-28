from sklearn import linear_model
from sklearn import neighbors
from sklearn import svm
import numpy as np


class TraditionalLearning:

    def trainLR(self,trainingDataSet):
        X = trainingDataSet[:,:-1]
        Y = trainingDataSet[:,-1]
        Y = np.reshape(Y, (np.shape(Y)[0], 1))

        clfLR = linear_model.SGDClassifier(loss='log', penalty='l2', verbose=1, warm_start=False, learning_rate='optimal')

        clfLR.fit(X, Y)
        return clfLR


    def trainSVM(self,trainingDataSet,C_value):
        X = trainingDataSet[:,:-1]
        Y = trainingDataSet[:,-1]
        Y = np.reshape(Y, (np.shape(Y)[0], 1))

        #clfSVM = linear_model.SGDClassifier(loss='hinge', penalty='l2', verbose=1,
        #                                n_iter=1, warm_start=False)

        #clfSVM.fit(X, Y)

        clfSVM_SVC = svm.SVC(C=C_value, kernel='rbf', probability=True, max_iter=100,
                             decision_function_shape='ovr',random_state=None)
        clfSVM_SVC.fit(X, Y)

        return clfSVM_SVC


    def trainKNN(self,trainingDataSet,k):
        X = trainingDataSet[:, :-1]
        Y = trainingDataSet[:, -1]
        Y = np.reshape(Y, (np.shape(Y)[0], 1))

        clfKNN = neighbors.KNeighborsClassifier(n_neighbors=k)
        clfKNN.fit(X, Y)
        return clfKNN

