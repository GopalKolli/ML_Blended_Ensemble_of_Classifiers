'''
@author = ru1saha
Learning Method class
This will have a couple of methods which we will add as we go on.

Method BlendClassification::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        Input Type   Argument Name  Definition
Input ; string      classifierName  Classifier Name     :  Name of the classifier
        integer     foldVal         CV fold value       :  Number of CV folds
        list        probabVal       Probability values  :  see def below
                                    This will be a list of numpy arrays
                                    where each of the element in the list
                                    will be a 2D array

                                    ArrayEle1 = This will be a 12 * num_training_examples
                                                where the first row will be the predicted
                                                probability values of all the examples
                                                to belong to class 1 (we have 12 classes)

Output; list        parameterVal    Parameter Value     :   List of parameter values
        numpy array currProbabVal   Current Probability Values  :   Probability values for the
                                    current classifier, same as above it will by a numpy
                                    array.
        float       blendRMSE       BlendRMSE           :   Blend RMSE value with the
                                    classifier if it converges.
'''

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from TraditionalLearning import TraditionalLearning
from sklearn import linear_model
from sklearn import neighbors
from sklearn import svm

class LearningMethod():

    def __init__(self,datasetname):
        self.trainingDataset = pd.read_csv(datasetname, sep = ',')
        self.trainingDataset = pd.DataFrame.as_matrix(self.trainingDataset)
        self.trainingDataset = self.trainingDataset[1:, 2:]
        self.trainingDataset = self.trainingDataset.astype(float)

        np.random.shuffle(self.trainingDataset)
        #check initialization of all datasets. they should be complete here.
        #And check wherever we are slicing arrays and lists. Lists and Arrays start at 0 index
        self.trainingLabels = self.trainingDataset[:,-1]
        self.trainingLabels = np.reshape(self.trainingLabels,(np.shape(self.trainingLabels)[0],1))
        #plt.ion()
        self.xaxes = []
        self.yaxes = []

        #self.trainingFeatures = self.trainingDataset[1:,3:-1]
        #self.trainingDataset = self.trainingDataset[:-1]

    def showPlots(self):
        for i in range(len(self.xaxes)):
            plt.plot(self.xaxes[i],self.yaxes[i])
        plt.show()

    def blendLearnClassificationLR(self, folds, probabVal):
        print("INSIDE blendLearnClassificationLR METHOD")
        row, col = np.shape(self.trainingDataset)
        probeSetSize = row / folds

        listOfParameterSets = []

        numberOfPreviousClassifiers = len(probabVal)
        if(len(probabVal) != 0):
            sumOfProbvals = np.sum(probabVal, axis=0)


        for i in range(folds):

            previousPredictionsToTakeAverage = []
            if(i==folds-1):
                probeset = self.trainingDataset[probeSetSize * (i):, :]
                trainSubSet = self.trainingDataset[:probeSetSize * (i), :]
                if(len(probabVal) != 0):
                    previousPredictionsToTakeAverage = sumOfProbvals[probeSetSize * (i):, :]
            else:
                probeset = self.trainingDataset[probeSetSize*(i):probeSetSize*(i+1),:]
                trainSubSet1 = self.trainingDataset[:probeSetSize * (i), :]
                trainSubSet2 = self.trainingDataset[probeSetSize * (i+1):, :]
                trainSubSet = np.concatenate((trainSubSet1,trainSubSet2),axis=0)
                if(len(probabVal) != 0):
                    previousPredictionsToTakeAverage = sumOfProbvals[probeSetSize*(i):probeSetSize*(i+1),:]

            #trainSubSet = self.trainingDataset - probeset
            X = trainSubSet[:,:-1]
            Y = trainSubSet[:,-1]
            Y = np.reshape(Y,(np.shape(Y)[0],1))

            #if (classifierName == 'LR'):
            converged = False
            previousRMSE = len(probeset)+20
            print("INITIAL RMSE : "+str(previousRMSE))
            clfLR = linear_model.SGDClassifier(loss='log', penalty='l2', verbose=1,
                                               n_iter=1, warm_start=False, learning_rate='optimal')

            clfLR.fit(X,Y)
            initCoef = clfLR.coef_
            initIntercept = clfLR.intercept_

            current_iteration_number = 0
            iteration_number_list = []
            RMSE_list = []
            while(not converged):
                current_iteration_number = current_iteration_number + 1
                iteration_number_list.append(current_iteration_number)
                #if (current_iteration_number == 1000):
                #    converged = True
                clfLR = linear_model.SGDClassifier(loss = 'log', penalty = 'l2', verbose = 1,
                                               n_iter = 1, warm_start = True, learning_rate='optimal')
                clfLR.fit(X,Y,initCoef,initIntercept)

                initCoef = clfLR.coef_
                initIntercept = clfLR.intercept_

                predictionsOnProbeSet = clfLR.predict_proba(probeset[:,:-1])
                predictionsOnProbeSet = np.nan_to_num(predictionsOnProbeSet)
                #print("PREDICTIONS ON PROBE SET", predictionsOnProbeSet[0])
                #print(predictionsOnProbeSet[1])

                if(not(numberOfPreviousClassifiers == 0)):
                    averageofBlendPredictions = np.sum([predictionsOnProbeSet,previousPredictionsToTakeAverage],axis=0)\
                                                /(numberOfPreviousClassifiers+1)
                else:
                    averageofBlendPredictions = predictionsOnProbeSet

                rownumber = 0
                sumsquares = 0
                probeSetTrueLabels = probeset[:,-1]
                probeSetTrueLabels = np.reshape(probeSetTrueLabels,(np.shape(probeSetTrueLabels)[0],1))
                for res in averageofBlendPredictions:
                    copyres = res.copy()
                    copyres[int(probeSetTrueLabels[rownumber])] = 1- copyres[int(probeSetTrueLabels[rownumber])]
                    sumsquares = sumsquares + math.pow(np.sum(copyres),2)
                    rownumber = rownumber + 1
                sumsquaresmean = sumsquares/np.shape(averageofBlendPredictions)[0]
                #print("SUMSQUARES", sumsquares)
                currentRMSE = math.sqrt(sumsquaresmean)
                #print("CURRENT RMSE BEFORE ERROR", currentRMSE)
                print("RMSE : "+str(currentRMSE))
                RMSE_list.append(currentRMSE)

                #percentChaneInRMSE=(previousRMSE-currentRMSE)*100/previousRMSE
                #print("percentChaneInRMSE : "+str(percentChaneInRMSE))

                if(currentRMSE>previousRMSE):
                    continue
                else:
                    percentChaneInRMSE = (previousRMSE - currentRMSE) * 100 / previousRMSE
                    if percentChaneInRMSE<=1:
                        converged = True
                        listOfParameterSets.append(clfLR)
                        print("FOR THIS PROBE SET- RMSE : " + str(currentRMSE))
                        self.xaxes.append(iteration_number_list)
                        self.yaxes.append(RMSE_list)
                        #plt.plot(iteration_number_list,RMSE_list)
                        #plt.title('probe'+str(i))
                        #plt.show()

                previousRMSE = currentRMSE
            print("ITERATIONS FOR THIS PROBE : "+str(current_iteration_number))

        predictionsOfThisClassifier = listOfParameterSets[0].predict_proba(self.trainingDataset[:, :-1])
        predictionsOfThisClassifier = np.nan_to_num(predictionsOfThisClassifier)
        for clfier in listOfParameterSets[1:]:
            predictionsOfThisClassifier = np.sum([predictionsOfThisClassifier,np.nan_to_num(clfier.predict_proba(self.trainingDataset[:, :-1]))],axis=0)
        predictionsOfThisClassifier = predictionsOfThisClassifier/len(listOfParameterSets)

        predictionsOfThisClassifierAvgWithPrevious = predictionsOfThisClassifier

        if (not (numberOfPreviousClassifiers == 0)):
            for prevProbval in probabVal:
                predictionsOfThisClassifierAvgWithPrevious = np.sum([predictionsOfThisClassifierAvgWithPrevious,prevProbval],axis=0)
            predictionsOfThisClassifierAvgWithPrevious = predictionsOfThisClassifierAvgWithPrevious/(len(probabVal)+1)

        rownumber = 0
        sumsquares = 0
        for finalresult in predictionsOfThisClassifierAvgWithPrevious[:]:
            copyfinalresult = finalresult.copy()
            copyfinalresult[int(self.trainingLabels[rownumber])] = 1 - copyfinalresult[int(self.trainingLabels[rownumber])]
            sumsquares = sumsquares + math.pow(np.sum(copyfinalresult), 2)
            rownumber = rownumber + 1
        sumsquaresmean = sumsquares / np.shape(predictionsOfThisClassifierAvgWithPrevious)[0]
        finalBlendRMSE = math.sqrt(sumsquaresmean)
        print("FINAL BLEND RMSE FOR LR : "+str(finalBlendRMSE))

        return [listOfParameterSets,predictionsOfThisClassifier,finalBlendRMSE]


    def blendLearnClassificationKNN(self, folds, probabVal):
        print("INSIDE blendLearnClassificationKNN METHOD")
        row, col = np.shape(self.trainingDataset)
        probeSetSize = row / folds

        listOfParameterSets = []

        numberOfPreviousClassifiers = len(probabVal)
        if (len(probabVal) != 0):
            sumOfProbvals = np.sum(probabVal,axis=0)

        for i in range(folds):
            previousPredictionsToTakeAverage = []
            if (i == folds - 1):
                probeset = self.trainingDataset[probeSetSize * (i):, :]
                trainSubSet = self.trainingDataset[:probeSetSize * (i), :]
                if (len(probabVal) != 0):
                    previousPredictionsToTakeAverage = sumOfProbvals[probeSetSize * (i):, :]
            else:
                probeset = self.trainingDataset[probeSetSize * (i):probeSetSize*(i + 1), :]
                trainSubSet1 = self.trainingDataset[:probeSetSize * (i), :]
                trainSubSet2 = self.trainingDataset[probeSetSize * (i + 1): , :]
                trainSubSet = np.concatenate((trainSubSet1,trainSubSet2),axis=0)
                if (len(probabVal) != 0):
                    previousPredictionsToTakeAverage = sumOfProbvals[probeSetSize * (i):probeSetSize*(i + 1), :]

            #trainSubSet = self.trainingDataset - probeset
            X = trainSubSet[:, :-1]
            Y = trainSubSet[:, -1]
            Y = np.reshape(Y,(np.shape(Y)[0],1))

            # if (classifierName == 'LR'):
            converged = False
            #previousRMSE = len(probeset) + 20
            #clfLR = linear_model.SGDClassifier(loss='log', penalty='l2', verbose=1,
            #                                   n_iter=1, warm_start=False)

            #clfLR.fit(X, Y)
            #initCoef = clfLR.coef_
            #initIntercept = clfLR.intercept_

            kvalues1 = [2,4,6,10,15,20,30,40,50,60,70,80,90,100]
            kvalues2 = [150,160,170,200,210,250,300,350,400,450,500,550,600,650]
            rmseValues1 = []
            rmseValues2 = []


            kindex = 0
            kset = 1
            k=0
            current_iteration_number = 0
            while (not converged):
                current_iteration_number = current_iteration_number + 1
                if kset == 1:
                    k = kvalues1[kindex]
                elif kset == 2:
                    k = kvalues2[kindex]
                kindex = kindex+1
                if kindex == 14 and kset == 1:
                    if rmseValues1.index(min(rmseValues1)) == 10:
                        kindex = 0
                        kset = 2
                        continue
                    else :
                        converged = True
                        cfltobestored = neighbors.KNeighborsClassifier(n_neighbors=kvalues1[rmseValues1.index(min(rmseValues1))])
                        cfltobestored.fit(X, Y)
                        listOfParameterSets.append(cfltobestored)
                        print("FOR THIS PROBE SET RMSE : " + str(min(rmseValues1)))
                        continue
                elif kindex == 14 and kset == 2:
                    converged = True
                    cfltobestored = neighbors.KNeighborsClassifier(
                        n_neighbors=kvalues2[rmseValues2.index(min(rmseValues2))])
                    cfltobestored.fit(X, Y)
                    listOfParameterSets.append(cfltobestored)
                    print("FOR THIS PROBE SET RMSE : " + str(min(rmseValues2)))
                    continue

                #set weights parameter to distance for distance based weighing
                clfKNN = neighbors.KNeighborsClassifier(n_neighbors = k)
                clfKNN.fit(X, Y)

                #initCoef = clfLR.coef_
                #initIntercept = clfLR.intercept_

                predictionsOnProbeSet = clfKNN.predict_proba(probeset[:, :-1])
                predictionsOnProbeSet = np.nan_to_num(predictionsOnProbeSet)
                #here we need multi class prediction probability values from KNNPredictions method
                #predictionsOnProbeSet = KNNImplementation.KNNPredictions(X,Y,probeset[:, :-1],k)

                if (not (numberOfPreviousClassifiers == 0)):
                    #check below
                    averageofBlendPredictions = np.sum([predictionsOnProbeSet,previousPredictionsToTakeAverage],axis=0) \
                                                / (numberOfPreviousClassifiers + 1)
                else:
                    averageofBlendPredictions = predictionsOnProbeSet

                rownumber = 0
                sumsquares = 0
                probeSetTrueLabels = probeset[:, -1]
                probeSetTrueLabels = np.reshape(probeSetTrueLabels,(np.shape(probeSetTrueLabels)[0],1))
                for res in averageofBlendPredictions:
                    copyres = res.copy()
                    copyres[int(probeSetTrueLabels[rownumber])] = 1 - copyres[int(probeSetTrueLabels[rownumber])]
                    sumsquares = sumsquares + math.pow(np.sum(copyres), 2)
                    rownumber = rownumber + 1
                sumsquaresmean = sumsquares / np.shape(averageofBlendPredictions)[0]
                currentRMSE = math.sqrt(sumsquaresmean)

                if kset == 1:
                    #rmseValues1[kindex] = currentRMSE
                    rmseValues1.append(currentRMSE)
                elif kset == 2:
                    #rmseValues2[kindex] = currentRMSE
                    rmseValues2.append(currentRMSE)

                #percentChaneInRMSE = (previousRMSE - currentRMSE) * 100 / previousRMSE
                #if (percentChaneInRMSE <= 0.01):
                #    converged = True
                #    listOfParameterSets.append(k)

                #previousRMSE = currentRMSE
            print("ITERATIONS FOR THIS PROBE : " + str(current_iteration_number))



        predictionsOfThisClassifier = listOfParameterSets[0].predict_proba(self.trainingDataset[0:probeSetSize, :-1])
        predictionsOfThisClassifier = np.nan_to_num(predictionsOfThisClassifier)

        j=1
        for clfier in listOfParameterSets[1:]:
            if (j == folds - 1):
                predictionsOfThisClassifier = np.concatenate((predictionsOfThisClassifier,np.nan_to_num(clfier.predict_proba(self.trainingDataset[probeSetSize * (j):, :-1]))),axis=0)
            else:
                predictionsOfThisClassifier = np.concatenate((predictionsOfThisClassifier,np.nan_to_num(clfier.predict_proba(self.trainingDataset[probeSetSize * (j):probeSetSize*(j + 1), :-1]))),axis=0)
            j = j+1


        #predictionsOfThisClassifier = predictionsOfThisClassifier / len(listOfParameterSets)

        predictionsOfThisClassifierAvgWithPrevious = predictionsOfThisClassifier

        if (not (numberOfPreviousClassifiers == 0)):
            for prevProbval in probabVal:
                predictionsOfThisClassifierAvgWithPrevious = np.sum([predictionsOfThisClassifierAvgWithPrevious,prevProbval],axis=0)
            predictionsOfThisClassifierAvgWithPrevious = predictionsOfThisClassifierAvgWithPrevious / (len(probabVal) + 1)


        rownumber = 0
        sumsquares = 0
        for finalresult in predictionsOfThisClassifierAvgWithPrevious:
            copyfinalresult = finalresult.copy()
            copyfinalresult[int(self.trainingLabels[rownumber])] = 1 - copyfinalresult[int(self.trainingLabels[rownumber])]
            sumsquares = sumsquares + math.pow(np.sum(copyfinalresult), 2)
            rownumber = rownumber + 1
        sumsquaresmean = sumsquares / np.shape(predictionsOfThisClassifierAvgWithPrevious)[0]
        finalBlendRMSE = math.sqrt(sumsquaresmean)
        print("FINAL BLEND RMSE FOR KNN : "+str(finalBlendRMSE))

        return [listOfParameterSets, predictionsOfThisClassifier, finalBlendRMSE]


    def blendLearnClassificationSVM(self, folds, probabVal):
        print("INSIDE blendLearnClassificationSVM METHOD")
        #from sklearn.kernel_approximation import (Nystroem)

        #featureSetTraining = self.trainingDataset[:, :-1]
        #labelsTraining = np.reshape(self.trainingDataset[:, -1], (np.shape(self.trainingDataset[:, -1])[0], 1))

        #featureNystroem = Nystroem(gamma=0.2, random_state=1)

        #trainingDatasetNew = featureNystroem.fit_transform(featureSetTraining, labelsTraining)

        row, col = np.shape(self.trainingDataset)
        probeSetSize = row / folds


        listOfParameterSets = []

        numberOfPreviousClassifiers = len(probabVal)
        if (len(probabVal) != 0):
            sumOfProbvals = np.sum(probabVal,axis=0)

        #EXPLODE DTATASET HERE USING SCIKIT KERNELIZATION FUNCTION

        for i in range(folds):
        #ALL DTATSETS FROM HERE CHANGES ACCORDING TO NEW DATASET CREATED ABOVE
            previousPredictionsToTakeAverage = []
            if (i == folds - 1):
                probeset = self.trainingDataset[probeSetSize * (i):, :]
                trainSubSet = self.trainingDataset[:probeSetSize * (i), :]
                if (len(probabVal) != 0):
                    previousPredictionsToTakeAverage = sumOfProbvals[probeSetSize * (i):, :]
            else:
                probeset = self.trainingDataset[probeSetSize * (i):probeSetSize*(i + 1), :]
                trainSubSet1 = self.trainingDataset[:probeSetSize * (i), :]
                trainSubSet2 = self.trainingDataset[probeSetSize * (i + 1):, :]
                trainSubSet = np.concatenate((trainSubSet1, trainSubSet2), axis=0)
                if (len(probabVal) != 0):
                    previousPredictionsToTakeAverage = sumOfProbvals[probeSetSize * (i):probeSetSize*(i + 1), :]

            #trainSubSet = trainingDatasetNew - probeset
            X = trainSubSet[:, :-1]
            Y = trainSubSet[:, -1]
            Y = np.reshape(Y,(np.shape(Y)[0],1))

            # if (classifierName == 'LR'):
            converged = False
            previousRMSE = len(probeset) + 20
            print("INITIAL RMSE : "+str(previousRMSE))

            #clfSVM = linear_model.SGDClassifier(loss='hinge', penalty='l2', verbose=1,
            #                                   n_iter=1, warm_start=False)

            #clfSVM.fit(X, Y)
            #initCoef = clfSVM.coef_
            #initIntercept = clfSVM.intercept_



            current_iteration_number = 0
            SVM_NumberOfIterations = 0
            while (not converged):
                current_iteration_number = current_iteration_number + 1
                SVM_NumberOfIterations = SVM_NumberOfIterations + 1

                #clfSVM = linear_model.SGDClassifier(loss='hinge', penalty='l2', verbose=1,
                #                                   n_iter=1, warm_start=True)
                #clfSVM.fit(X, Y, initCoef, initIntercept)

                #initCoef = clfSVM.coef_
                #initIntercept = clfSVM.intercept_
                clfSVM_SVC = svm.SVC(C=10.0, kernel='rbf', probability=True, max_iter=SVM_NumberOfIterations, decision_function_shape='ovr',random_state=None)
                clfSVM_SVC.fit(X, Y)


                predictionsOnProbeSet = clfSVM_SVC.predict_proba(probeset[:, :-1])
                predictionsOnProbeSet = np.nan_to_num(predictionsOnProbeSet)

                if (not (numberOfPreviousClassifiers == 0)):
                    averageofBlendPredictions = np.sum([predictionsOnProbeSet,previousPredictionsToTakeAverage],axis=0)/(numberOfPreviousClassifiers + 1)
                else:
                    averageofBlendPredictions = predictionsOnProbeSet

                rownumber = 0
                sumsquares = 0
                probeSetTrueLabels = probeset[:, -1]
                probeSetTrueLabels = np.reshape(probeSetTrueLabels,(np.shape(probeSetTrueLabels)[0],1))
                for res in averageofBlendPredictions:
                    copyres = res.copy()
                    copyres[int(probeSetTrueLabels[rownumber])] = 1 - copyres[int(probeSetTrueLabels[rownumber])]
                    sumsquares = sumsquares + math.pow(np.sum(copyres), 2)
                    rownumber = rownumber + 1
                sumsquaresmean = sumsquares / np.shape(averageofBlendPredictions)[0]
                currentRMSE = math.sqrt(sumsquaresmean)

                #percentChaneInRMSE = (previousRMSE - currentRMSE) * 100 / previousRMSE
                #if (percentChaneInRMSE <= 0.01):
                #    converged = True
                #    listOfParameterSets.append(clfSVM_SVC)
                #    print("RMSE FOR THIS FOLD : " + str(currentRMSE))

                if (currentRMSE > previousRMSE):
                    continue
                else:
                    percentChaneInRMSE = (previousRMSE - currentRMSE) * 100 / previousRMSE
                    if percentChaneInRMSE <= 2:
                        converged = True
                        listOfParameterSets.append(clfSVM_SVC)
                        SVM_NumberOfIterations = 0
                        print("FOR THIS PROBE SET- RMSE : " + str(currentRMSE))

                previousRMSE = currentRMSE
            print("ITERATIONS FOR THIS PROBE : " + str(current_iteration_number))

        predictionsOfThisClassifier = listOfParameterSets[0].predict_proba(self.trainingDataset[:, :-1])
        predictionsOfThisClassifier = np.nan_to_num(predictionsOfThisClassifier)
        for clfier in listOfParameterSets[1:]:
            predictionsOfThisClassifier = np.sum([predictionsOfThisClassifier,np.nan_to_num(clfier.predict_proba(self.trainingDataset[:, :-1]))],axis=0)
        predictionsOfThisClassifier = predictionsOfThisClassifier / len(listOfParameterSets)

        predictionsOfThisClassifierAvgWithPrevious = predictionsOfThisClassifier

        if (not (numberOfPreviousClassifiers == 0)):
            for prevProbval in probabVal:
                predictionsOfThisClassifierAvgWithPrevious = np.sum([predictionsOfThisClassifierAvgWithPrevious,prevProbval],axis=0)
            predictionsOfThisClassifierAvgWithPrevious = predictionsOfThisClassifierAvgWithPrevious / (len(probabVal) + 1)

        rownumber = 0
        sumsquares = 0
        for finalresult in predictionsOfThisClassifierAvgWithPrevious[:]:
            copyfinalresult = finalresult.copy()
            copyfinalresult[int(self.trainingLabels[rownumber])] = 1 - copyfinalresult[int(self.trainingLabels[rownumber])]
            sumsquares = sumsquares + math.pow(np.sum(copyfinalresult), 2)
            rownumber = rownumber + 1
        sumsquaresmean = sumsquares / np.shape(predictionsOfThisClassifierAvgWithPrevious)[0]
        finalBlendRMSE = math.sqrt(sumsquaresmean)
        print("FINAL BLEND RMSE FOR SVM : "+str(finalBlendRMSE))

        return [listOfParameterSets, predictionsOfThisClassifier, finalBlendRMSE]



    def predictProbablitiesLR(self,clf,inputFeatureSet):
        return np.nan_to_num(clf.predict_proba(inputFeatureSet))

    def predictProbablitiesSVM(self,clf,inputFeatureSet):
        return np.nan_to_num(clf.predict_proba(inputFeatureSet))

    def predictProbablitiesKNN(self,clf,inputFeatureSet):
        return np.nan_to_num(clf.predict_proba(inputFeatureSet))


    def predictLR(self,clf,inputFeatureSet):
        return np.nan_to_num(clf.predict(inputFeatureSet))

    def predictSVM(self,clf,inputFeatureSet):
        return np.nan_to_num(clf.predict(inputFeatureSet))

    def predictKNN(self,clf,inputFeatureSet):
        return np.nan_to_num(clf.predict(inputFeatureSet))

    def learnLR(self):
        trlearn = TraditionalLearning()
        return trlearn.trainLR(self.trainingDataset)

    def learnSVM(self,c_value):
        trlearn = TraditionalLearning()
        return trlearn.trainSVM(self.trainingDataset,c_value)

    def learnKNN(self,k):
        trlearn = TraditionalLearning()
        return trlearn.trainKNN(self.trainingDataset,k)