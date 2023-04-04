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

  def modelEval(self):
    print("\n--- model.evaluate() ---")
    print("Using x_train + y_train (%i): "%(x_train.shape[0]))
    self.tfModel.model.evaluate(x_train,  y_train, verbose=2)
    print("***")
    print("Using x_test + y_test (%i): "%(x_test.shape[0]))
    self.tfModel.model.evaluate(x_test,  y_test, verbose=2)

  def build(self, paramDict):
    self.dense1 = paramDict['dense1']
    self.dropout1 = paramDict['dropout1']
    self.trainingNum = paramDict['trainingNum']

    model = self.tfModel.createModel(dense1=self.dense1, dropout1=self.dropout1, 
                                    x_trainSet=x_train, y_trainSet=y_train)
    self.modelEval()
    self.probability_model = self.tfModel.getProbabilityModel(model)
    
  def run(self):
    # 16/1/23 DH: 'x_test' is a 3-D array of 10,000 28*28 images
    self.imgNum = x_test.shape[0]
    f = open("predicted-errors.txt", "w")

    softmax2DList = self.probability_model(x_test).numpy()
    softmaxList = softmax2DList[0]
    print("\nSoftmax list for element 0 of 'x_test': ",softmaxList )

    print("\nLooping through",self.imgNum,"images from x_test")
    self.iCnt = 0
    errorNum = 0

    for elem in range(self.imgNum):
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
    self.gspreadErrors.updateSheet(sheet,2,9,"ooh yea...")
    self.gspreadErrors.addRow(sheet, dense=self.dense1, dropout=self.dropout1, 
                              training_num=self.trainingNum, test_num=self.imgNum, errors=self.iCnt)
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

  # 1/4/23 DH: List of dicts for DNN params
  paramDictList = [
    {'dense1': 784, 'dropout1': None, 'trainingNum': x_train.shape[0], 'runs': 1, 'reruns': 1 },
    ]

  tfCfg.batchRunAshore(paramDictList)