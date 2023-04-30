import matplotlib.pyplot as plt

# 30/3/23 DH: Refactor of model creation + DB model access
from tf_model import *
from gspread_errors import *
# 30/4/23 DH: Refactor of GSpreadErrors class
from gspread_rl import *
from gspread_rl_parts import *

"""
Load and prepare the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). 
Convert the sample data from integers to floating-point numbers 
("The training set contains 60000 examples, and the test set 10000 examples")
"""
# http://yann.lecun.com/exdb/mnist/
mnist = tf.keras.datasets.mnist
# https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

class TFConfig(object):

  def __init__(self) -> None:
    self.tfModel = TFModel()
    self.gspreadErrors = GSpreadErrors(spreadsheet="Addresses",sheet="mnist-errors")
    # 30/4/23 DH: Refactor GSpreadErrors class
    self.gspreadRL = GSpreadRL(spreadsheet="Addresses",sheet="mnist-rl")
    self.gspreadRLparts = GSpreadRLparts(spreadsheet="Addresses",sheet="mnist-rl-parts")

  def displayImg(self,elem):
    # https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html
    plt.imshow(x_test[elem], cmap='gray_r')
    
    # 22/1/23 DH: Calling function without '()' does not return an error but does NOT execute it (like function pointer)
    plt.draw()
    plt.waitforbuttonpress(timeout=1)

  def modelEval(self,start=False):
    print("--- model.evaluate() ---")
    
    #print("Using x_train + y_train (%i): "%(x_train.shape[0]))
    #self.tfModel.model.evaluate(x_train,  y_train, verbose=2)
    #print("***")
    
    print("Using x_test + y_test (%i): "%(x_test.shape[0]))
    evalRes = self.tfModel.model.evaluate(x_test,  y_test, verbose=2)
    accuracyPercent = "{:.2f}".format(evalRes[1])
    print("evaluate() accuracy:",accuracyPercent)
    print("------------------------\n")

    if start == True:
      self.startPercent = accuracyPercent
      # 28/4/23 DH: Lowest percent gets lowered as appropriate during the retraining
      self.lowestPercent = accuracyPercent
      self.accuracies = []

    if hasattr(self, 'accuracies'):
      self.accuracies.append(accuracyPercent)

    return accuracyPercent

  def build(self, paramDict):
    self.dense1 = paramDict['dense1']
    self.dropout1 = paramDict['dropout1']
    self.trainingNum = paramDict['trainingNum']
    # 24/4/23 DH:
    self.x_trainSet = paramDict['x_trainSet']
    self.y_trainSet = paramDict['y_trainSet']
    # 23/4/23 DH:
    self.epochs = paramDict['epochs']

    #print("x_train:",type(self.x_trainSet),self.x_trainSet.shape )
    #print("y_train:",type(self.y_trainSet),self.y_trainSet.shape )

    self.model = self.tfModel.createTrainedModel(dense1=self.dense1, dropout1=self.dropout1,
                                            x_trainSet=self.x_trainSet, y_trainSet=self.y_trainSet, 
                                            epochs=self.epochs)
    self.modelEval(start=True)
  
  # 28/4/23 DH:
  def checkBreakout(self):
    # 27/4/23 DH: Get an updated 'softmax2DList' after a specified '% increase' in accuracy
    if hasattr(self, 'accuracyPercent'):
      # 28/4/23 DH:
      if float(self.lowestPercent) > float(self.accuracyPercent):
        self.lowestPercent = self.accuracyPercent

      partDict = self.runPartNumbers[self.runPartNum]

      if float(partDict['lowestPercent']) > float(self.accuracyPercent):
        partDict['lowestPercent'] = self.accuracyPercent
      
      if float(partDict['highestPercent']) < float(self.accuracyPercent):
        partDict['highestPercent'] = self.accuracyPercent

      # 28/4/23 DH:
      if float(self.accuracyPercent) > float(self.startPercent) + self.desiredIncrease:
        partDict['endPercent'] = self.accuracyPercent
        return True
      
    # END: ------------- 'if hasattr(self, 'accuracyPercent')' ---------------

    return False

  # 27/4/23 DH:
  def rlRunPart(self):
    # 23/4/23 DH: Get trained NN (wrapped with softmax layer)
    self.probability_model = self.tfModel.getProbabilityModel(self.model)
    softmax2DList = self.probability_model(x_test).numpy()

    self.runPartNum += 1
    
    """
    # 28/4/23 DH: Now each part is a dictionary (within the 'runPartNumbers' dictionary):
      startPercent (DONE)
      endPercent (DONE)
      lowestPercent (DONE)
      highestPercent (DONE)

      partStartCnt (DONE)
    """
    self.runPartNumbers[self.runPartNum] = {'partStartCnt': self.iCnt}

    self.imgNum = x_test.shape[0]
    print("*************************************************************************")
    print(self.runPartNum,") Looping through",self.imgNum,"images from x_test")
    print("*************************************************************************\n")
    
    for elem in range(self.imgNum):
      predictedVal = np.argmax(softmax2DList[elem])
      if y_test[elem] != predictedVal:
        self.iCnt += 1

        # 22/4/23 DH: Send 'x_test[elem]' to 'bitwiseAndDefinitive()' to get self checking update
        # (...which in this case is a little REDUNDANT since 'y_test[elem]' is the answer we need...!)

        # 26/4/23 DH: Adding 'class weights' didn't help (prob due to overriding TF algorithms)
        classWeightDict = {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1}
        classWeightDict[ y_test[elem] ] = 2

        x_test_elemArray = np.array([x_test[elem]])
        y_test_elemArray = np.array([y_test[elem]])

        #print("x_test[elem]:",type(x_test_elemArray),x_test_elemArray.shape )
        #print("y_test[elem]:",type(y_test_elemArray),y_test_elemArray.shape )
        print("Predicted value:",predictedVal,", Expected value:",y_test[elem])

        # 26/4/23 DH: 'train_on_batch' resulted in "tensor 'Placeholder/_1' value" error
        self.tfModel.model.fit(x=x_test_elemArray, y=y_test_elemArray)

        self.accuracyPercent = self.modelEval()
        if not 'startPercent' in self.runPartNumbers[self.runPartNum]:
          print("Adding startPercent ", self.accuracyPercent,"to part",self.runPartNum,"\n")
          self.runPartNumbers[self.runPartNum]['startPercent'] = self.accuracyPercent
          self.runPartNumbers[self.runPartNum]['lowestPercent'] = self.accuracyPercent
          self.runPartNumbers[self.runPartNum]['highestPercent'] = self.accuracyPercent

        if self.checkBreakout():
          break

      # END: ------------- 'if y_test[elem] != predictedVal' --------------

      if self.errorNum != self.iCnt and self.iCnt % 100 == 0:
        print("####################################################################")
        print(self.iCnt, "errors at element",elem)
        print("####################################################################")
        print()
        self.errorNum = self.iCnt

    # END: ------------- 'for elem in range(self.imgNum)' -------------
  
  # 29/4/23 DH:
  def getPartCnt(self):
    # 28/4/23 DH: Print part count (counts recorded at start for each part, not num in part)
    partCnt = 0
    currentPart = self.runPartNumbers[self.key]

    if self.key + 1 in self.runPartNumbers:

      nextPart = self.runPartNumbers[self.key + 1]
      self.subCnt = nextPart['partStartCnt']

      partCnt = nextPart['partStartCnt'] - currentPart['partStartCnt']
    else:
      self.subCnt = self.iCnt - self.subCnt
      partCnt = self.subCnt
    
    return partCnt

  # 28/4/23 DH:
  def printPartStats(self):
    # 28/4/23 DH: Added in for stats debug
    for key in self.runPartNumbers.keys():
      print(key,":",self.runPartNumbers[key])
    print()

    self.subCnt = 0

    for self.key in self.runPartNumbers.keys():

      partCnt = self.getPartCnt()
      print(self.key,":",partCnt)

      # 28/4/23 DH: Other metrics for part
      currentPart = self.runPartNumbers[self.key]

      start = currentPart['startPercent']
      end = currentPart['endPercent']
      low = currentPart['lowestPercent']
      high = currentPart['highestPercent']

      print("  : (start:",start,", end:",end,", lowest:",low,", highest:",high,")")
    # END: ------- 'for key in self.runPartNumbers.keys()' -------

  # 29/4/23 DH:
  def printStats(self):
    print("-----------")
    print("Total errors:",self.iCnt)
    print("Run parts:",self.runPartNum)
    print("Accuracy start :",self.startPercent)
    print("Accuracy end   :",self.accuracyPercent)
    print("Lowest accuracy:",self.lowestPercent)
    print()
    self.printPartStats()
    print()
    print(self.accuracies)

  def populateGSheetRLparts(self):
    sheet = self.gspreadRLparts.sheet
    
    self.subCnt = 0

    for self.key in self.runPartNumbers.keys():

      partCnt = self.getPartCnt()

      # 28/4/23 DH: Other metrics for part
      currentPart = self.runPartNumbers[self.key]
      
      partStart = currentPart['startPercent']
      partEnd = currentPart['endPercent']
      partLow = currentPart['lowestPercent']
      partHigh = currentPart['highestPercent']

      # Date,Test number,Part number,Count,Start,End,Lowest,Highest

      dateOfEntry = self.gspreadRL.dateOfEntry
      testnum = self.gspreadRL.testnum
      
      self.gspreadRLparts.addRowRLparts(sheet, entry_date=dateOfEntry, test_num=testnum, part_num=self.key,
        count=partCnt, start=partStart, end=partEnd, lowest=partLow, highest=partHigh)                 

    # END: ------- 'for key in self.runPartNumbers.keys()' -------

  # 29/4/23 DH:
  def populateGSheetRL(self):
    self.gspreadErrors.updateSheet(self.gspreadErrors.sheet, 2, 10, "ooh yea...")

    sheet = self.gspreadRL.sheet
    
    self.gspreadRL.addRowRL(sheet, dense=self.dense1, dropout=self.dropout1, 
      training_num=self.trainingNum, retrain_num=self.iCnt, run_parts=self.runPartNum,
      accuracy_start=self.startPercent, accuracy_end=self.accuracyPercent, lowest_accuracy=self.lowestPercent)

    self.populateGSheetRLparts()

    self.gspreadRL.getGSheetsData(sheet)

  # 24/4/23 DH:
  def rlRun(self, paramDictList):
    
    for paramDict in paramDictList:

      self.build(paramDict)

      self.iCnt = 0
      self.errorNum = 0
      self.runPartNum = 0
      # 28/4/23 DH: Now a dictionary of dictionaries
      self.runPartNumbers = {}

      self.desiredIncrease = 0.05
      self.rlRunPart()
      while float(self.accuracyPercent) < 0.50:
        self.desiredIncrease += 0.05
        self.rlRunPart()

      self.printStats()
      # 29/4/23 DH:
      self.populateGSheetRL()

  def run(self):
    # 16/1/23 DH: 'x_test' is a 3-D array of 10,000 28*28 images
    self.imgNum = x_test.shape[0]
    f = open("predicted-errors.txt", "w")

    # 23/4/23 DH: Get trained NN (wrapped with softmax layer)
    self.probability_model = self.tfModel.getProbabilityModel(self.model)

    # 22/4/23 DH: Add ALL TEST DATA to trained model (wrapped with softmax layer)
    # (to update the predicted output from flattened image will prob require disappearing into TensorFlow...)
    softmax2DList = self.probability_model(x_test).numpy()
    softmaxList = softmax2DList[0]
    print("\nSoftmax list for element 0 of 'x_test': ",softmaxList )

    print("\nLooping through",self.imgNum,"images from x_test")
    self.iCnt = 0
    errorNum = 0

    for elem in range(self.imgNum):
      # 25/4/23 DH: https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict
      predictedVal = np.argmax(softmax2DList[elem])
      if y_test[elem] != predictedVal:
        f.write("Dataset Element: "+ str(elem) + " Expected: "+ str(y_test[elem]) + " Predicted: " + str(predictedVal) + "\n")
        self.iCnt += 1

      if errorNum != self.iCnt and self.iCnt % 100 == 0:
        print(self.iCnt, "errors at element",elem)
        errorNum = self.iCnt

      lastNum = elem

    print("-----------")
    print("Last element: ",lastNum)
    print("Total errors: ",self.iCnt)

    f.write("-----------\n")
    f.write("Last element: " + str(lastNum) + "\n")
    f.write("Total errors: " + str(self.iCnt) + "\n")
    f.close()

  def populateGSheet(self, paramDict):
    # 27/3/23 DH: Now add the results of the errors to gsheet
    sheet = self.gspreadErrors.sheet
    self.gspreadErrors.updateSheet(sheet,2,10,"ooh yea...")
    self.gspreadErrors.addRow(sheet, dense=self.dense1, dropout=self.dropout1, 
                              training_num=self.trainingNum, test_num=self.imgNum, 
                              epochs=self.epochs, errors=self.iCnt)
    # 31/3/23 DH: Add "=average()" in appropriate G row for last row of a DNN build

    self.gspreadErrors.getGSheetsData(sheet)

  # 1/4/23 DH:
  def batchRunAshore(self, paramDictList):
    for paramDict in paramDictList:
      for run in range(paramDict['runs']):
        self.build(paramDict)

        for rerun in range(paramDict['reruns']):
          self.run()
          self.populateGSheet(paramDict)


# =============================================================================================

# 30/3/23 DH:
if __name__ == '__main__':
  tfCfg = TFConfig()

  # 24/4/23 DH:
  x_trainSet = x_train[:700]
  y_trainSet = y_train[:700]
  #x_trainSet = x_train[:2000]
  #y_trainSet = y_train[:2000]

  # 1/4/23 DH: List of dicts for DNN params
  paramDictList = [
    #{'dense1': 784, 'dropout1': None, 'trainingNum': x_train.shape[0], 'epochs': 5, 'runs': 1, 'reruns': 1 },
    
    {'dense1': 20, 'dropout1': None, 'epochs': 1, 'x_trainSet': x_trainSet, 'y_trainSet': y_trainSet,
     'trainingNum': x_trainSet.shape[0], 'runs': 1, 'reruns': 1 },
    ]
  
  # 29/4/23 DH: Need cmd line arg:
  #  train
  #  rl

  #tfCfg.batchRunAshore(paramDictList)

  # 24/4/23 DH:
  tfCfg.rlRun(paramDictList)