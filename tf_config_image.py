# 12/5/23 DH:

from tf_config import *
from tf_config_misc import *

import numpy

class TFConfigImage(TFConfig):

  def __init__(self) -> None:
    super().__init__()

    # 15/5/23 DH: Needed for 'saveDigitArrayValues()'
    self.tfCfgMisc = TFConfigMisc()

    self.createDigitLabels()

  # ---------------------------- API --------------------------------

  def createImages(self, display=False, number=10):
    imgs = self.x_test[:number]
    imgValues = self.y_test[:number]

    x_testPlusDigit = []

    # 'shape' returns a tuple so access first value
    imgNum = imgs.shape[0]
    print("imgNum:",imgNum)
    #imgNum = 1
    for elem in range(imgNum):
      img = imgs[elem]

      # TODO: Add a pixel value set for small expected number onto the original image to create 'y_test'
      #       Then train the 784-784 DNN to do that for any 'x_test' to create 'x_testPlusDigit'

      if elem < 10:
        print("y_test[",elem,"]:",imgValues[elem])

      if elem == 11:
        print("...")
      
      img = img * 255
      img = np.asarray(img, dtype = 'uint8')

      img = self.addDigitToImage(img, imgValues[elem])
      # Test each of the prepared digits
      #img = self.addDigitToImage(img, elem)

      if display:
        #self.displayImg(img,timeout=-1)
        self.displayImg(img,timeout=1)
      
      x_testPlusDigit.append(img)
    # END: ---- 'for elem in range(imgNum)' ----

    self.x_testPlusDigit = numpy.asarray(x_testPlusDigit)

    # Now train the 784-784 DNN with 'self.x_test' + 'self.x_testPlusDigit'

  def runImageTrainer(self, paramDict):
    self.build(paramDict)

    self.probability_model = self.tfModel.getProbabilityModel(self.model)

    # Put in unaltered MNIST digit
    #        ---------
    softmax2DList = self.probability_model(self.x_test).numpy()
    imgList = self.tfModel.model(self.x_test).numpy()

    for elem in range(self.x_testPlusDigit.shape[0]):
      imgSoftmax = softmax2DList[elem]
      imgSoftmax = imgSoftmax.reshape(28,28)

      imgModel = imgList[elem]
      imgModel = imgModel.reshape(28,28)

      print("Displaying altered MNIST image")
      self.displayImg(self.x_testPlusDigit[elem],timeout=3)

      print("Displaying probability_model() image from orig image")
      self.displayImg(imgSoftmax,timeout=3)
      
      print("Displaying model() image from orig image")
      self.displayImg(imgModel,timeout=3)
      
      print("Displaying orig MNIST image")
      self.displayImg(self.x_test[elem],timeout=3)

      if elem < 2:
        self.tfCfgMisc.saveDigitArrayValues(imgSoftmax, fileNum=elem)
        self.tfCfgMisc.saveDigitArrayValues(imgModel, fileNum=100+elem)
        
  # ---------------------------- Internal -------------------------------
  
  # Override the 'TFConfig.build(paramDict)'
  def build(self, paramDict):
    
    self.trainingNum = paramDict['trainingNum']
    self.x_trainSet = paramDict['x_trainSet']
    self.y_trainSet = paramDict['y_trainSet']
    self.epochs = paramDict['epochs']

    self.y_trainSet = self.y_trainSet.reshape(self.y_trainSet.shape[0],28*28)

    # 14/5/23 DH: 
    print("self.x_trainSet:",type(self.x_trainSet),self.x_trainSet.shape)
    print("self.y_trainSet:",type(self.y_trainSet),self.y_trainSet.shape)
    print("self.y_test:",type(self.y_test),self.y_test.shape)

    self.model = self.tfModel.createTrainedImageModel(x_trainSet=self.x_trainSet, 
                                                      y_trainSet=self.y_trainSet, 
                                                      epochs=self.epochs)

    self.modelEval(start=True)


  def createDigitLabels(self):
    self.digitLabelsDict = {}

    # 6 * 7 = 42 pixels
    # 13/5/23 DH: Add the digit graphic to the MNIST image to create MNIST training result image (ie 'y_test')
    #             784-784 DNN
    self.digitLabelsDict[0] = numpy.asarray(
                              [[  0,  0,255,255,  0,  0],
                               [  0,255,255,255,255,  0],
                               [255,255,  0,  0,255,255],
                               [255,  0,  0,  0,  0,255],
                               [255,255,  0,  0,255,255],
                               [  0,255,255,255,255,  0],
                               [  0,  0,255,255,  0,  0]])
    
    self.digitLabelsDict[1] = numpy.asarray(
                              [[  0,255,255,255,  0,  0],
                               [  0,255,255,255,  0,  0],
                               [  0,  0,  0,255,  0,  0],
                               [  0,  0,  0,255,  0,  0],
                               [  0,  0,  0,255,  0,  0],
                               [  0,255,255,255,255,  0],
                               [  0,255,255,255,255,  0]])
    
    self.digitLabelsDict[2] = numpy.asarray(
                              [[  0,  0,255,255,  0,  0],
                               [  0,255,  0,255,255,  0],
                               [  0,  0,  0,  0,255,  0],
                               [  0,  0,  0,  0,255,  0],
                               [  0,  0,  0,255,255,  0],
                               [  0,  0,255,255,  0,  0],
                               [  0,255,255,255,255,255]])
    
    self.digitLabelsDict[3] = numpy.asarray(
                              [[  0,  0,255,255,255,  0],
                               [  0,255,  0,  0,255,255],
                               [  0,  0,  0,  0,  0,255],
                               [  0,  0,  0,255,255,  0],
                               [  0,  0,  0,  0,  0,255],
                               [  0,255,  0,  0,255,255],
                               [  0,  0,255,255,255,  0]])
    
    self.digitLabelsDict[4] = numpy.asarray(
                              [[  0,  0,  0,255,255,  0],
                               [  0,  0,255,  0,255,  0],
                               [  0,255,  0,  0,255,  0],
                               [255,  0,  0,  0,255,  0],
                               [255,255,255,255,255,255],
                               [  0,  0,  0,  0,255,  0],
                               [  0,  0,  0,  0,255,  0]])
    
    self.digitLabelsDict[5] = numpy.asarray(
                              [[255,255,255,255,  0,  0],
                               [255,  0,  0,  0,  0,  0],
                               [255,255,255,255,  0,  0],
                               [  0,  0,  0,255,255,  0],
                               [  0,  0,  0,  0,255,  0],
                               [255,  0,  0,255,255,  0],
                               [255,255,255,255,  0,  0]])

    self.digitLabelsDict[6] = numpy.asarray(
                              [[  0,  0,255,255,  0,  0],
                               [  0,255,  0,  0,  0,  0],
                               [  0,255,  0,  0,  0,  0],
                               [  0,255,255,255,255,  0],
                               [  0,255,  0,  0,255,  0],
                               [  0,255,  0,  0,255,  0],
                               [  0,  0,255,255,255,  0]])

    self.digitLabelsDict[7] = numpy.asarray(
                              [[255,255,255,255,255,255],
                               [255,255,255,255,255,255],
                               [  0,  0,  0,  0,255,255],
                               [  0,  0,  0,255,255,  0],
                               [  0,  0,255,255,  0,  0],
                               [  0,255,255,  0,  0,  0],
                               [255,255,  0,  0,  0,  0]])
    
    self.digitLabelsDict[8] = numpy.asarray(
                              [[  0,  0,255,255,255,  0],
                               [  0,255,255,  0,255,255],
                               [  0,255,  0,  0,  0,255],
                               [  0,  0,255,255,255,  0],
                               [  0,255,  0,  0,  0,255],
                               [  0,255,255,  0,255,255],
                               [  0,  0,255,255,255,  0]])

    self.digitLabelsDict[9] = numpy.asarray(
                              [[  0,255,255,255,  0,  0],
                               [  0,255,  0,  0,255,  0],
                               [  0,255,  0,  0,255,  0],
                               [  0,  0,255,255,255,  0],
                               [  0,  0,  0,  0,255,  0],
                               [  0,  0,  0,  0,255,  0],
                               [  0,  0,  0,  0,255,  0]])


  def addDigitToImage(self, img, digit):
    xSize, ySize = img.shape

    #if not digit in self.digitLabelsDict:
    #digit = 3

    xLblSize, yLblSize = self.digitLabelsDict[digit].shape
    #print(xSize,ySize,",",xLblSize,yLblSize)
    imgLbl = self.digitLabelsDict[digit]

    for x in range(xSize):
      for y in range(ySize):
        if x < xLblSize and y < yLblSize:
          img[x+1][y+1] = imgLbl[x][y]

    """
    for x in range(xSize):
      xStr = ""
      for y in range(ySize):
        pixVal = "{:4}".format(img[x][y])
        xStr += str(pixVal)
      
      print(xStr)
    print()
    """

    return img
        