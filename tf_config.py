import matplotlib.pyplot as plt

# 30/3/23 DH: Refactor of model creation + DB model access
from tf_model import *
from gspread_errors import *

# 6/5/23 DH:
import pickle

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
    self.mnistFilename = "digitDictionary.pkl"

  def displayImg(self,imgDict,elem):
    # https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html
    plt.imshow(imgDict[elem], cmap='gray_r')
    
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

  # 5/5/23 DH: http://yann.lecun.com/exdb/mnist/
  """
  "SD-3 is much cleaner and easier to recognize than SD-1"
  "Therefore it was necessary to build a new database by mixing NIST's datasets."
  
  "The MNIST training set is composed of 30,000 patterns from SD-3 and 30,000 patterns from SD-1. 
  Our test set was composed of 5,000 patterns from SD-3 and 5,000 patterns from SD-1. "
  """
  def getMNISTexamples(self):
    print("Getting MNIST examples...")

    digitDict = {0:None, 1:None, 2:None, 3:None, 4:None, 5:None, 6:None, 7:None, 8:None, 9:None}

    imgs = x_test
    imgValues = y_test

    imgNum = imgs.shape[0]

    #print(digitDict.keys())
    #print(digitDict.values())

    # 5/5/23 DH: Need to find way to select a good image for each digit 
    #            (via a shortlist array for each digit which then gets selected during 'displayImg()')
    for elem in range(imgNum):
      digitDict[imgValues[elem]] = imgs[elem]
      
      if not any(x is None for x in digitDict.values()):
        print("Got full set at", elem)
        break
    
    for key in digitDict.keys():
      """
      if digitDict[key] is None:
        print(key,":",digitDict[key])
      else:
        print(key,":",type(digitDict[key]))
      """
    
      self.displayImg(digitDict, key)

    # "pickle rick"...
    with open(self.mnistFilename, 'wb') as fp:
      pickle.dump(digitDict, fp)

  def checkMNISTexamples(self):
    print("Checking MNIST examples in", self.mnistFilename)

    with open(self.mnistFilename, 'rb') as fp:
      digitDict = pickle.load(fp)

    for key in digitDict.keys():
      #print(key,":",type(digitDict[key]))

      self.displayImg(digitDict, key)
    
    
