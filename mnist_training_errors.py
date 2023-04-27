import matplotlib.pyplot as plt

# 30/3/23 DH: Refactor of model creation + DB model access
from tf_model import *
from gspread_errors import *

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
    self.gspreadErrors = GSpreadErrors()

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
    
  # 27/4/23 DH:
  def rlRunPart(self):
    # 23/4/23 DH: Get trained NN (wrapped with softmax layer)
    self.probability_model = self.tfModel.getProbabilityModel(self.model)
    softmax2DList = self.probability_model(x_test).numpy()

    self.runPartNum += 1
    self.runPartNumbers[self.runPartNum] = self.iCnt
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

      if self.errorNum != self.iCnt and self.iCnt % 100 == 0:
        print("####################################################################")
        print(self.iCnt, "errors at element",elem)
        print("####################################################################")
        print()
        self.errorNum = self.iCnt

      # 27/4/23 DH: Get an updated 'softmax2DList' after a 10% increase in accuracy
      if hasattr(self, 'accuracyPercent') and float(self.accuracyPercent) > float(self.startPercent) + self.desiredIncrease:
        break
    # END: 'for elem in range(self.imgNum)'

  # 24/4/23 DH:
  def rlRun(self, paramDictList):
    
    for paramDict in paramDictList:

      self.build(paramDict)

      self.iCnt = 0
      self.errorNum = 0
      self.runPartNum = 0
      self.runPartNumbers = {}

      self.desiredIncrease = 0.05
      self.rlRunPart()
      while float(self.accuracyPercent) < 0.60:
        self.desiredIncrease += 0.05
        self.rlRunPart()

      print("-----------")
      print("Total errors:",self.iCnt)
      print("Run parts:",self.runPartNum)
      
      print(self.runPartNumbers)
      subCnt = 0
      for key in self.runPartNumbers.keys():
        if key + 1 in self.runPartNumbers:
          print(key,":",self.runPartNumbers[key + 1] - self.runPartNumbers[key])
          subCnt = self.runPartNumbers[key + 1]
        else:
          subCnt = self.iCnt - subCnt
          print(key,":",subCnt)

      print("Accuracy start:",self.startPercent)
      print("Accuracy end  :",self.accuracyPercent)
      print(self.accuracies)

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
  #x_trainSet = x_train
  #y_trainSet = y_train

  # 1/4/23 DH: List of dicts for DNN params
  paramDictList = [
    #{'dense1': 784, 'dropout1': None, 'trainingNum': x_train.shape[0], 'epochs': 5, 'runs': 1, 'reruns': 1 },
    
    {'dense1': 20, 'dropout1': None, 'epochs': 1, 'x_trainSet': x_trainSet, 'y_trainSet': y_trainSet,
     'trainingNum': x_trainSet.shape[0], 'runs': 1, 'reruns': 1 },
    ]

  #tfCfg.batchRunAshore(paramDictList)

  # 24/4/23 DH:
  tfCfg.rlRun(paramDictList)