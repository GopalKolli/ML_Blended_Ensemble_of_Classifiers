from MainClass import MainClass
import matplotlib.pyplot as plt
import time

class RunClass:
    #mainObjLR = MainClass()
    #mainObjSVM = MainClass()
    #mainObjKNN = MainClass()
    #mainObj = MainClass()
    #xaxis = []
    #yaxis = []

    def run(self):

        #For Blended Ensemble

        #self.mainObj.blendLearn()
        #SET THESE PARAMETERS AFTER PARAMETER TUNING
        C_val = 1.0
        k_val = 2

        #self.mainObj.learnLR()
        #self.mainObj.learnSVM(C_val)
        #self.mainObj.learnKNN(k_val)

        #accuracy = self.mainObj.blendTest()
        #print("NORMAL ENSEMBLE WITHOUT WEIGHTED VOTING : " + str(accuracy))
        listofsizes = [1000,2000,4000,8000,16000,32000]
        listoftimes = []
        for dataset in ["First1k.csv", "First2k.csv", "First4k.csv", "First8k.csv", "First16k.csv", "First32k.csv"]:
            mainObj = MainClass(dataset)
            starttime = time.time()
            mainObj.blendLearnMethod()
            mainObj.blendLearn()
            endtime = time.time()
            diff = endtime-starttime
            print("DIFFERENCE IS : ",diff)
            listoftimes.append(diff)
        plt.plot(listofsizes,listoftimes)
        #self.mainObj.learnKNN(2)
        #self.mainObj.learnSVM()
        #lr = [0.001,0.01,0.1,1.0,10.0,100.0]
        #for i in lr:

#below is the code used for Parameter Tuning----------------------------------------
'''
        for cl in ["LR"]:
            if(cl == "LR"):
                #acc = []
                #self.xaxis.append([0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0])
                #for learn_rate in [0.00001,0.0001,0.001,0.01,0.1,1.0,10.0]:
                    LRObj = MainClass()
                    LRObj.learnLR()
                    accuracy = LRObj.blendTest()
                    #print("LOGISTIC_REGRESSION HYPERPARAMETER TUNING")
                    print("LOGISTIC_REGRESSION OPTIMAL LEARNING RATE")
                    #print()
                    print("Accuracy : "+str(accuracy))
                    #acc.append(accuracy)
                #self.yaxis.append(acc)


            elif (cl == "SVM"):
                #acc = []
                #self.xaxis.append([1.0,5.0,10.0,20.0,30.0,40.0,50.0,60.0,80.0,90.0,100.0])
                for c_value in [1.0,5.0,10.0,20.0]:
                    ObjSVM = MainClass()
                    ObjSVM.learnSVM(c_value)
                    accuracy = ObjSVM.blendTest()
                    print("SVM HYPERPARAMETER TUNING")
                    print(c_value)
                    print(accuracy)
                    #acc.append(accuracy)
                #self.yaxis.append(acc)

            elif (cl == "KNN"):
                #acc = []
                #self.xaxis.append([2, 4, 6, 10, 14, 18, 20, 30, 40, 50])
                for k in [2, 4, 6, 10, 14, 18, 20, 30, 40, 50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,220,250]:
                    ObjKNN = MainClass()
                    ObjKNN.learnKNN(k)
                    accuracy = ObjKNN.blendTest()
                    print("KNN HYPERPARAMETER TUNING")
                    print("k value: " + str(k))
                    print(accuracy)
                    #acc.append(accuracy)
                #self.yaxis.append(acc)


        #for i in range(len(self.xaxis)):
        #    plt.plot(self.xaxis[i], self.yaxis[i])
        #plt.show()
'''
# above is the code used for Parameter Tuning----------------------------------------

        #self.mainObj.learnLR()

        #accuracy = self.mainObj.blendTest()
        #print("Learning Rate: "+str(i))
        #print(accuracy)
        #self.mainObj.showPlots()

            #for k in [2, 4, 6, 10, 14, 20, 30, 40, 50]:
            #    self.mainObj.learnKNN(k)
            #    accuracy = self.mainObj.blendTest()
            #    print(k)
            #    print(accuracy)

r = RunClass()
r.run()