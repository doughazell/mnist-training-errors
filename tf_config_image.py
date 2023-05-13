# 12/5/23 DH:

from tf_config import *
from tf_config_misc import *

class TFConfigImage(TFConfig):

  def __init__(self) -> None:
    super().__init__()

    self.tfCfgMisc = TFConfigMisc()

    self.createDigitLabels()

  def createDigitLabels(self):
    self.digitLabelsDict = {}

    # 6 * 7 = 42 pixels
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
                               [  0,  0,  0,  0,255,255],
                               [  0,  0,  0,  0,  0,255],
                               [  0,  0,  0,255,255,  0],
                               [  0,  0,  0,  0,  0,255],
                               [  0,  0,  0,  0,255,255],
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

  def displayImages(self, number=10):
    imgs = self.x_test[:number]
    imgValues = self.y_test[:number]

    # 'shape' returns a tuple so access first value
    imgNum = imgs.shape[0]
    print("imgNum:",imgNum)
    #imgNum = 1
    for elem in range(imgNum):
      img = imgs[elem]

      # TODO: Add a pixel value set for small expected number onto the original image
      #       Then train the 784-784 DNN to do that for any 'x_test:y_test' pair

      print("y_test[",elem,"]:",imgValues[elem])
      
      img = img * 255
      img = np.asarray(img, dtype = 'uint8')

      img = self.addDigitToImage(img, imgValues[elem])
      # Test each of the prepared digits
      #img = self.addDigitToImage(img, elem)

      #self.displayImg(img,timeout=-1)
      self.displayImg(img,timeout=3)
      
      #self.tfCfgMisc.saveDigitArrayValues(img, fileNum=elem)

  def addDigitToImage(self, img, digit):
    xSize, ySize = img.shape

    #if not digit in self.digitLabelsDict:
    #digit = 2

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
        