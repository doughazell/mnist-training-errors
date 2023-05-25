# 30/4/23 DH: Refactor TFConfig class
from tf_config import *

# 30/4/23 DH: Refactor of GSpreadErrors class
from gspread_rl import *
from gspread_rl_parts import *

# 7/5/23 DH:
import signal
import sys

class TFConfigRL(TFConfig):
  def __init__(self, tfConfigTrain, integer=False) -> None:
    
    # Get access to parent attributes via 'super()'
    super().__init__(integer=integer)
    
    # 30/4/23 DH: Refactor GSpreadErrors class
    self.gspreadRL = GSpreadRL(spreadsheet="Addresses",sheet="mnist-rl")
    self.gspreadRLparts = GSpreadRLparts(spreadsheet="Addresses",sheet="mnist-rl-parts")

    self.tfConfigTrain = tfConfigTrain

    # 7/5/23 DH:
    signal.signal(signal.SIGINT, self.signal_handler)

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

  # 10/5/23 DH:
  def retrainToBreakout(self,elem):
    """
    # 7/5/23 DH: TFConfig.bitwiseAND() shows that self checking via bitwise-AND with example image
                  is only 25% accurate (7428/10000 errors).
    
    Demonstrates efficacy of TF
    
    1) 'y_test[elem]' not available for operational system, so need to try intermittent retrain 
    with random small training sets (like agent CPD...)
    2) Selective retrain EVERY FAILURE after batch training to 50% accurate
    """
    # 22/4/23 DH: Send 'x_test[elem]' to 'bitwiseAndDefinitive()' to get self checking update
    # (...which in this case is a little REDUNDANT since 'y_test[elem]' is the answer we need...!)

    # 26/4/23 DH: Adding 'class weights' didn't help (prob due to overriding TF algorithms)
    classWeightDict = {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1}
    classWeightDict[ self.y_test[elem] ] = 2

    x_test_elemArray = np.array([self.x_test[elem]])
    y_test_elemArray = np.array([self.y_test[elem]])

    # 26/4/23 DH: 'train_on_batch' resulted in "tensor 'Placeholder/_1' value" error
    #
    # 10/5/23 DH: TRAIN ON EVERY FAILURE
    self.tfModel.model.fit(x=x_test_elemArray, y=y_test_elemArray)

    self.recordAccuracy()

    if self.checkBreakout():
      return True

    return False # ie no breakout...

  # 10/5/23 DH:
  def recordAccuracy(self):
    self.accuracyPercent = self.modelEval()

    if not 'startPercent' in self.runPartNumbers[self.runPartNum]:
    
      print("Adding startPercent ", self.accuracyPercent,"to part",self.runPartNum,"\n")
    
      self.runPartNumbers[self.runPartNum]['startPercent'] = self.accuracyPercent
      self.runPartNumbers[self.runPartNum]['lowestPercent'] = self.accuracyPercent
      self.runPartNumbers[self.runPartNum]['highestPercent'] = self.accuracyPercent

  # 27/4/23 DH:
  def rlRunPart(self, rl):
    # 23/4/23 DH: Get trained NN (wrapped with softmax layer)
    self.probability_model = self.tfModel.getProbabilityModel(self.model)
    softmax2DList = self.probability_model(self.x_test).numpy()

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

    self.imgNum = self.x_test.shape[0]
    print("*************************************************************************")
    print(self.runPartNum,") Looping through",self.imgNum,"images from x_test")
    print("*************************************************************************\n")
    
    if rl == False:
      self.recordAccuracy()

    for elem in range(self.imgNum):
      predictedVal = np.argmax(softmax2DList[elem])
      
      if self.y_test[elem] != predictedVal:
        self.iCnt += 1
        #print("Predicted value:",predictedVal,", Expected value:",self.y_test[elem])

        if rl == True:
          if self.retrainToBreakout(elem) == True:
            break

      # END: ------------- 'if y_test[elem] != predictedVal' --------------

      if self.errorNum != self.iCnt and self.iCnt % 100 == 0:
        print("####################################################################")
        print(self.iCnt, "errors at element",elem)
        print("####################################################################")
        print()
        # "%100" error gets printed out ONLY ONCE (not until "%100 + 1" error)
        self.errorNum = self.iCnt

    # END: ------------- 'for elem in range(self.imgNum)' -------------
  
  # 24/4/23 DH:
  def rlRun(self, paramDictList, rl=True):
    
    for paramDict in paramDictList:

      self.build(paramDict)
      self.trgTotal = paramDict['trainingNum']

      self.iCnt = 0
      self.errorNum = 0
      self.runPartNum = 0
      # 28/4/23 DH: Now a dictionary of dictionaries
      self.runPartNumbers = {}

      # 10/5/23 DH: Running the 'rl' command (and not 'cpd') 
      if rl == True:
        self.desiredIncrease = 0.05
        self.rlRunPart(rl)

        while float(self.accuracyPercent) < 0.90:
          
          self.desiredIncrease += 0.05
          self.rlRunPart(rl)
      
        self.printStats()
        # 29/4/23 DH:
        self.populateGSheetRL()
      
      # 10/5/23 DH: Running the 'cpd' command 
      else:
        self.rlRunPart(rl)

        # 'accuracyPercent' rounded to 2 decimal places in 'TFConfig.modelEval()' so can reach 1.0, ie 100%
        # (Needs < 51 errors of 10,000 in 'self.rlRunPart()' for 2dp to round to 100%)
        while float(self.accuracyPercent) < 1.0:
          
          self.tfModel.model.fit(x=self.x_test, y=self.y_test)
          x_testNum = self.x_test.shape[0]
          self.trgTotal += x_testNum

          # Now also train with non tested images
          """
          10,10 = 50/50
          20,10 = 66/33
          30,10 = 75/25
          40,10 = 80/20
          Result =  760700 for 50/50 images cf 260700 to reach 2dp rounded 100%
          Result = 1950700 for 66/33 images
          Result = 2040700 for 75/25 images
          Result = 4750700 for 80/20 images
          """
          ratio = 0
          if ratio > 0:
            self.tfModel.model.fit(x=self.x_train[:(x_testNum * ratio)], y=self.y_train[:(x_testNum * ratio)])
            self.trgTotal += (x_testNum * ratio)

          self.rlRunPart(rl)
        
        self.printStats()
  
  # 7/5/23 DH: Handling interrupt when RL does not run to completion (due to Spaceport exception handling)
  #            ...get the stats gained before stopping to coach a debrief.
  def signal_handler(self, sig, frame):
    print('\nYou pressed Ctrl+C so saving stats (to coach a debrief)...')

    self.printStats()

    print("\nTFConfigRL.signal_handler()")
    print("  #self.populateGSheetRL()")
    #self.populateGSheetRL()
    
    sys.exit(0)

  # ========================= Display stats + populate Google Sheets =====================
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

      if 'startPercent' in currentPart:
        start = currentPart['startPercent']
      else:
        start = "XXX"

      # 7/5/23 DH: Needed for Ctrl-C interrupt handling
      if 'endPercent' in currentPart: 
        end = currentPart['endPercent']
      else:
        end = "XXX"
      
      if 'lowestPercent' in currentPart:
        low = currentPart['lowestPercent']
      else:
        low = "XXX"

      if 'highestPercent' in currentPart:
        high = currentPart['highestPercent']
      else:
        high = "XXX"

      print("  : (start:",start,", end:",end,", lowest:",low,", highest:",high,")")
    # END: ------- 'for key in self.runPartNumbers.keys()' -------

  # 29/4/23 DH:
  def printStats(self):
    print("-----------")
    print("Training total:",self.trgTotal)
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

      # 7/5/23 DH: Needed for Ctrl-C interrupt handling
      if 'endPercent' in currentPart:
        partEnd = currentPart['endPercent']
      else:
        partEnd = "XXX"
      
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


  
