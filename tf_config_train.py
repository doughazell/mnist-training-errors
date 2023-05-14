# 30/4/23 DH: Refactor TFConfig class
from tf_config import *
from gspread_errors import *

class TFConfigTrain(TFConfig):

  def __init__(self, integer=False) -> None:
    
    # Get access to parent attributes via 'super()'
    super().__init__(integer=integer)
    if integer:
      self.misc = "Integer images"
    else:
      self.misc = None

    self.gspreadErrors = GSpreadErrors(spreadsheet="Addresses",sheet="mnist-errors")

  def run(self):
    # 16/1/23 DH: 'x_test' is a 3-D array of 10,000 28*28 images
    self.imgNum = self.x_test.shape[0]
    f = open("predicted-errors.txt", "w")

    # 23/4/23 DH: Get trained NN (wrapped with softmax layer)
    self.probability_model = self.tfModel.getProbabilityModel(self.model)

    # 22/4/23 DH: Add ALL TEST DATA to trained model (wrapped with softmax layer)
    # (to update the predicted output from flattened image will prob require disappearing into TensorFlow...)
    softmax2DList = self.probability_model(self.x_test).numpy()
    
    softmaxList = softmax2DList[0]
    print("\nSoftmax list for element 0 of 'x_test': ",softmaxList )

    print("\nLooping through",self.imgNum,"images from x_test")
    self.iCnt = 0
    errorNum = 0

    for elem in range(self.imgNum):
      # 25/4/23 DH: https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict
      predictedVal = np.argmax(softmax2DList[elem])
      if self.y_test[elem] != predictedVal:
        f.write("Dataset Element: "+ str(elem) + " Expected: "+ str(self.y_test[elem]) + " Predicted: " + str(predictedVal) + "\n")
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
                              epochs=self.epochs, errors=self.iCnt, misc=self.misc)
    
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
