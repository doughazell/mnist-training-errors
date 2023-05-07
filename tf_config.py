import matplotlib.pyplot as plt

# 30/3/23 DH: Refactor of model creation + DB model access
from tf_model import *
from gspread_errors import *

# 6/5/23 DH:
import pickle
import time
import numpy

"""
Load and prepare the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). 
Convert the sample data from integers to floating-point numbers (WHY...???)

("The training set contains 60000 examples, and the test set 10000 examples")
"""
mnist = tf.keras.datasets.mnist
# https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 6/5/23 DH: WHY...????
#  '$ tf-test train' appears to be more accurate for floats vs integers (5800 vs 7600 for 20Dense-700Train)
x_train, x_test = x_train / 255.0, x_test / 255.0

class TFConfig(object):

  def __init__(self) -> None:
    self.tfModel = TFModel()

    self.mnistFilename = "digitDictionary.pkl"
    self.mnistFilenameINT = "digitDictionaryINTEGER.pkl"

    with open(self.mnistFilenameINT, 'rb') as fp:
      self.digitDict = pickle.load(fp)

  # 6/5/23 DH: 'self.digitDict' used to be single image per digit key (rather than a list)
  def displayDictImg(self,imgDict,elem):
    self.displayImg(imgDict[elem])

  # 6/5/23 DH: If a key/mouse is pressed on an image then it goes into prune mode
  def displayImgList(self,imgList):
    newImgList = []

    for index, img in enumerate(imgList):
      notification = self.displayImg(img)

      if notification is not None:
        print(index)
        newImgList.append(img)
    
    if len(newImgList) > 0:
      return True, newImgList
    
    return False, imgList
  
  def displayImg(self,img, timeout=1):
    # https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html
    plt.imshow(img, cmap='gray_r')
    
    # 22/1/23 DH: Calling function without '()' does not return an error but does NOT execute it (like function pointer)
    plt.draw()
    retButtonPress = plt.waitforbuttonpress(timeout=timeout)
    # 'True' if key press, 'False' if mouse press, 'None' if timeout
    return retButtonPress
    

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

    #digitDict = {0:None, 1:None, 2:None, 3:None, 4:None, 5:None, 6:None, 7:None, 8:None, 9:None}
    self.digitDict = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}

    imgs = x_test
    imgValues = y_test

    imgNum = imgs.shape[0]

    #print(digitDict.keys())
    #print(digitDict.values())

    # 5/5/23 DH: Need to find way to select a good image for each digit 
    #            (via a shortlist array for each digit which then gets selected in 'displayImgList()')
    """
    self.digitDict:
      0 ..........
      1 ..........
      2 ..........
      ...
    """
    for elem in range(imgNum):
      #digitDict[imgValues[elem]] = imgs[elem]
      self.digitDict[imgValues[elem]].append(imgs[elem])
      
      # 6/5/23 DH: Change this from 'not None' to a 'list of 10' per elem
      #if not any(x is None for x in digitDict.values()):
      if not any(len(list) < 10 for list in self.digitDict.values()):
        print("Got full set at", elem)
        break
    
    self.pruneDigitDict(printOut=True)

    # "pickle rick"...https://www.youtube.com/watch?v=_gRnvDRFYN4
    with open(self.mnistFilename, 'wb') as fp:
      pickle.dump(self.digitDict, fp)

    for key in self.digitDict.keys():
      #self.displayDictImg(digitDict, key)
      self.displayImgList(self.digitDict[key])

  def pruneDigitDict(self, printOut=False):
    for key in self.digitDict.keys():
      if printOut:
        print(key,":",len(self.digitDict[key]))
      
      # 6/5/23 DH: Now limit size to 10 elements
      self.digitDict[key] = self.digitDict[key][:10]

  def checkMNISTexamples(self, digit=None):
    print("Checking MNIST examples in", self.mnistFilename)
    changed = False

    if digit:
      #self.displayDictImg(digitDict, int(digit))
      changed, self.digitDict[int(digit)] = self.displayImgList(self.digitDict[int(digit)])

    else:

      for key in self.digitDict.keys():

        #self.displayDictImg(digitDict, key)
        chgFlag, self.digitDict[key] = self.displayImgList(self.digitDict[key])
        
        # Prevent mid list changes being forgotten for repickling
        if chgFlag:
          changed = True
    
    if changed:
      print("List shortened so repickling")
      with open(self.mnistFilename, 'wb') as fp:
        pickle.dump(self.digitDict, fp)

  # --------------------------------------------------------------------------------------------
  # 7/5/23 DH: Test bitwise-AND with example images for reinforcement learning
  # --------------------------------------------------------------------------------------------
  def printZeroDigitArrayValues(self):
    # Get the image for dictionary key '0'
    testImg = self.digitDict[0][0]

    print("Test img:",type(testImg), testImg.shape, type(testImg[0][0]))

    xOffset = 15
    y = 15
    print(xOffset,",",y,"of",testImg.shape)
    for xIdx in range(7):
      xIdx += xOffset

      # 6/5/23 DH: When remove "Convert the sample data from integers to floating-point numbers"
      #            "{:.2f}" becomes "{:3}"
      print("{:3}".format(testImg[xIdx][y]),
            "{:3}".format(testImg[xIdx][y+1]),
            "{:3}".format(testImg[xIdx][y+2]),
            "{:3}".format(testImg[xIdx][y+3]),
            "{:3}".format(testImg[xIdx][y+4]),
            "{:3}".format(testImg[xIdx][y+5]),
            "{:3}".format(testImg[xIdx][y+6]),
            "{:3}".format(testImg[xIdx][y+7]))
      
      """
      print(testImg[xIdx][y],
            testImg[xIdx][y+1],
            testImg[xIdx][y+2],
            testImg[xIdx][y+3],
            testImg[xIdx][y+4],
            testImg[xIdx][y+5],
            testImg[xIdx][y+6],
            testImg[xIdx][y+7])
      """
      
    print()

  def getImgCheckTotals(self, img):
    #totals = []
    totalsDict = {}

    for key in self.digitDict.keys():
  
      testImg = self.digitDict[key][0]

      bitwiseAndRes = numpy.bitwise_and(testImg, img)
      imgX, imgY = bitwiseAndRes.shape
      #print("Shape:",imgX, imgY)

      iTotal = 0
      for x in range(imgX):
        for y in range(imgY):
          iTotal += bitwiseAndRes[x][y]

      #print(key,"total:", iTotal)
      #totals.append(iTotal)
      totalsDict[key] = iTotal

    return totalsDict
  
  def bitwiseAND(self):
    print("Correlating 'y_test[index]' with return of highest bitwise-AND\n")

    #self.printZeroDigitArrayValues()
    
    # 7/5/23 DH: 'x_test' is converted to float above (for NN accuracy reasons)
    imgs = x_test
    imgValues = y_test

    imgNum = imgs.shape[0]
    #imgNum = 2

    totalsErrors = 0
    errorNum = 0
    for elem in range(imgNum):

      # Convert image from float to integer array
      img = imgs[elem]
      img = img * 255
      img = np.asarray(img, dtype = 'uint8')
      
      totalsDict = self.getImgCheckTotals(img)

      totals = list(totalsDict.values())
      digit = np.argmax(totals)

      if int(digit) is not int(imgValues[elem]):
        totalsErrors += 1
        #self.displayImg(img, timeout=2)
      
      if totalsErrors < 3:
        print("Check:", digit, "y_test:",imgValues[elem])
        for k in sorted(totalsDict, key=totalsDict.get, reverse=True):
          if k == imgValues[elem]:
            print("*",k,"-",totalsDict[k])
          else:
            print(k,"-",totalsDict[k])
        print()
      """
      else:
        print("Breaking after 3 errors for debug")
        break
      """

      if errorNum != totalsErrors and totalsErrors % 100 == 0:
        print(totalsErrors, "errors at element",elem)
        errorNum = totalsErrors

    # ----- END: 'for elem in range(imgNum)' -----

    print("Total errors:", totalsErrors)
    
  # 7/5/23 DH:
  def convertDigitDict(self):
    print("Converting example digits from float to integer")

    for key in self.digitDict.keys():
      # The example digits only contain 1 element per digit list
      img = self.digitDict[key][0]

      # 6/5/23 DH: Reverse the float conversion after 'mnist.load_data()'
      img = img * 255
      img = np.asarray(img, dtype = 'uint8')

      self.digitDict[key][0] = img

    with open(self.mnistFilenameINT, 'wb') as fp:
        pickle.dump(self.digitDict, fp)
  
  # --------------------------------------------------------------------------------------------
