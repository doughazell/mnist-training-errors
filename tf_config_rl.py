# 30/4/23 DH: Refactor TFConfig class
from tf_config import *

# 30/4/23 DH: Refactor of GSpreadErrors class
from gspread_rl import *
from gspread_rl_parts import *

class TFConfigRL(TFConfig):
  def __init__(self, tfConfigTrain) -> None:
    
    # Get access to parent attributes via 'super()'
    super().__init__()
    
    # 30/4/23 DH: Refactor GSpreadErrors class
    self.gspreadRL = GSpreadRL(spreadsheet="Addresses",sheet="mnist-rl")
    self.gspreadRLparts = GSpreadRLparts(spreadsheet="Addresses",sheet="mnist-rl-parts")

    self.tfConfigTrain = tfConfigTrain

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
    self.tfConfigTrain.gspreadErrors.updateSheet(self.tfConfigTrain.gspreadErrors.sheet, 
                                                 2, 10, "ooh yea...")

    sheet = self.gspreadRL.sheet
    
    self.gspreadRL.addRowRL(sheet, dense=self.dense1, dropout=self.dropout1, 
      training_num=self.trainingNum, retrain_num=self.iCnt, run_parts=self.runPartNum,
      accuracy_start=self.startPercent, accuracy_end=self.accuracyPercent, lowest_accuracy=self.lowestPercent)

    self.populateGSheetRLparts()

    # 1/5/23 DH: Overriden parent class 'getGSheetsData()' in 'GSpreadRL'
    #self.gspreadRL.getGSheetsData(sheet)
    # 1/5/23 DH: Access parent class 'getGSheetsData()'
    #super(type(self.gspreadRL), self.gspreadRL).getGSheetsData(sheet)

    print()
    self.gspreadRL.getGSheetsDataRL(sheet, self.gspreadRLparts)

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
  
