import math
import collections
import numpy as np
import pandas as pd
import sklearn
from BlendLearning import LearningMethod
from TraditionalLearning import TraditionalLearning

#Remember that this is multiclass classification. This applies everywhere.
#Careful about the variables and their types, dimensions

class MainClass():
    #thetraindataset=""
    def __init__(self,traindataset):
        self.thetraindataset = traindataset
        self.testSet = pd.read_csv('TestHoSet.csv', sep = ',')
        self.testSet = pd.DataFrame.as_matrix(self.testSet)
        self.testSet = self.testSet[1:, 4:]
        #self.testSet = self.testSet[1:, 3:]
        self.testSet = self.testSet.astype(float)

        np.random.shuffle(self.testSet)
        self.testSetInputs = self.testSet[:,:-1]
        self.testSetOutputs = self.testSet[:,-1]
        self.testSetOutputs = np.reshape(self.testSetOutputs, (np.shape(self.testSetOutputs)[0], 1))

    classifiers = collections.OrderedDict()

    #classifiers['KNN'] = []
    classifiers['SVM'] = []
    classifiers['LR'] = []

    #classifiers['RF'] = []

    folds = 4
    probvals = []
    blendRMSE = math.inf
    blendRMSEOrder = []

    #testSet = np.array
    #dim = testSet.shape
    #testSetInputs = np.array
    #testSetOutputs = np.array

    def blendLearnMethod(self):
        self.blendLearnObj = LearningMethod(self.thetraindataset)
    #blendLearnMethod()
    #def __init__(self):
    #load test set from csv file.

    def blendLearn(self):
        for classifier in self.classifiers.keys():
            print("CURRENT CLASSIFIER IS : " + classifier)

            if(classifier == "LR"):
                result = self.blendLearnObj.blendLearnClassificationLR(self.folds,self.probvals)
            elif(classifier == "SVM"):
                result = self.blendLearnObj.blendLearnClassificationSVM(self.folds,self.probvals)
            elif(classifier == "KNN"):
                result = self.blendLearnObj.blendLearnClassificationKNN(self.folds, self.probvals)

            currentparameters = result[0]
            currentprobvals = result[1]
            currentblendRMSE = result[2]
            print("BLEND RMSE OBTAINED BY BLENDING IN " + classifier + " IS " ,currentblendRMSE)


            if(currentblendRMSE<=self.blendRMSE):
                print("CLASSIFIER ACCEPTED")
                self.probvals.append(currentprobvals)
                self.classifiers[classifier] = currentparameters
                self.blendRMSEOrder.append(currentblendRMSE)
                self.blendRMSE = currentblendRMSE
            else:
                print("CLASSIFIER REJECTED")
                # will the classifiers order be preserved properly?
                #del self.classifiers[classifier]
                continue

    def blendTest(self):
        # predict the prob values using 5 sets of parameters of a classifier and take averages to get final predictions from that classifier
        # DO this for all classifiers
        # Average the reslut of all classifiers to get the final predictions.
        # Using the above results, claculate RMSE and return the value

        allpredictions = []

        for classifier in self.classifiers.keys():
            if classifier=="LR":
                parametersSets = self.classifiers[classifier]
                if(parametersSets != []):
                    for currentparametersSet in parametersSets:
                        predictedProbs = self.blendLearnObj.predictProbablitiesLR(currentparametersSet,self.testSetInputs)
                        #predictedProbs = predictedProbs*(34/(34+58+74))
                        allpredictions.append(predictedProbs)
            elif classifier == "SVM":
                #from sklearn.kernel_approximation import (Nystroem)

                #featureNystroem = Nystroem(gamma=0.2, random_state=1)

                #kernelizedTestingDataset = featureNystroem.fit_transform(self.testSetInputs, self.testSetOutputs)
                #kernelizedTestingDatasetInputs = kernelizedTestingDataset[:,:-1]

                parametersSets = self.classifiers[classifier]
                if (parametersSets != []):
                    for currentparametersSet in parametersSets:
                        predictedProbs = self.blendLearnObj.predictProbablitiesSVM(currentparametersSet, self.testSetInputs)
                        #predictedProbs = predictedProbs * (58 / (34 + 58 + 74))
                        allpredictions.append(predictedProbs)
            elif classifier == "KNN":
                parametersSets = self.classifiers[classifier]
                if (parametersSets != []):
                    for currentparametersSet in parametersSets:
                        predictedProbs = self.blendLearnObj.predictProbablitiesKNN(currentparametersSet, self.testSetInputs)
                        #predictedProbs = predictedProbs * (74 / (34 + 58 + 74))
                        allpredictions.append(predictedProbs)
            #SIMILAR ELIFs....

        predsum = np.sum(allpredictions,axis=0)
        #set the dimensions (1xm) and all zeros
        #for aprediction in allpredictions:
            #predsum = np.sum([predsum,aprediction],axis=0)

        averagePredictions = predsum/len(allpredictions)


        finalPedictionsList = []
        rowindex = 0
        for resultRow in averagePredictions:
            resultList = np.argsort(resultRow)
            predictedClassForThisRow = resultList[-1]
            finalPedictionsList.append(predictedClassForThisRow)
            rowindex = rowindex+1
        totalNumberOfPredictions = len(finalPedictionsList)
        finalPedictions = np.reshape(np.array(finalPedictionsList),(totalNumberOfPredictions,1))

        numberOfMatchedPredictions = np.sum(finalPedictions == self.testSetOutputs)

        accuracyPercent = (numberOfMatchedPredictions/totalNumberOfPredictions)*100
        #diff = finalPedictions-self.testsetOutputs
        #diffsquared = np.power(diff,2)
        #diffsquaredmean = np.mean(diffsquared)
        #testRMSE = np.sqrt(diffsquaredmean)
        print("ACCURACY OF THIS BLENDED ENSEMBLE OF CLASSIFIERS IS "+str(accuracyPercent)+"%")
        return accuracyPercent


    def learnLR(self):
        blendLearnObjLR = LearningMethod()
        self.classifiers['LR'] = [blendLearnObjLR.learnLR()]

    def learnSVM(self,C_value):
        blendLearnObjLR = LearningMethod()
        self.classifiers['SVM'] = [blendLearnObjLR.learnSVM(C_value)]

    def learnKNN(self,k):
        blendLearnObjLR = LearningMethod()
        self.classifiers['KNN'] = [blendLearnObjLR.learnKNN(k)]


    def showPlots(self):
        self.blendLearnObj.showPlots()