# 12/5/23 DH: Refactor TFConfig for:
"""
API
---
mnist
  getMNISTexamples()
    pruneDigitDict()

  checkMNISTexamples()

bitwise
  bitwiseAND()
    getImgCheckTotals()
      printDigitArrayValues()
      saveDigitArrayValues()
      
  convertDigitDict()

"""

from tf_config import *

# 6/5/23 DH:
import pickle
import time
import numpy
import sys

class TFConfigMisc(TFConfig):

  def __init__(self) -> None:
    super().__init__()

    self.mnistFilename = "digitDictionary.pkl"
    self.mnistFilenameINT = "digitDictionaryINTEGER.pkl"
    self.digitFilename = "digitOutput"

    with open(self.mnistFilenameINT, 'rb') as fp:
    #with open(self.mnistFilename, 'rb') as fp:
      self.digitDict = pickle.load(fp)
    
  # =========================== API =================================
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

    imgs = self.x_test
    imgValues = self.y_test

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
  #           (which is an example of "Expert system" and shows power of AI for image variation)
  # --------------------------------------------------------------------------------------------
  def bitwiseAND(self, check=False):
    print("Correlating 'y_test[index]' with return of highest bitwise-AND\n")

    testImg = self.digitDict[0][0]
    self.printDigitArrayValues(testImg)
    
    # 7/5/23 DH: 'x_test' is converted to float above (for NN accuracy reasons)
    if check:
      # 'digitDict.values()' is array of 1 elem arrays
      valuesList = list(self.digitDict.values())
      # imgs = numpy.asarray(valuesList[0])
      
      # Need to loop through the first element in each of the digit array
      #imgs = np.ndarray(0)
      imgList = []
      for key in self.digitDict.keys():
        #imgs = np.append(imgs, valuesList[key])
        imgList.append(valuesList[key][0])
      imgs = numpy.asarray(imgList)

      imgValues = list(self.digitDict.keys())
    else:
      imgs = self.x_test
      imgValues = self.y_test

    totalsErrors = 0
    errorNum = 0
    
    # 'shape' returns a tuple so access first value
    imgNum = imgs.shape[0]
    print("imgNum:",imgNum)
    #imgNum = 1
    for elem in range(imgNum):

      img = imgs[elem]

      totalsDict = self.getImgCheckTotals(img)

      totals = list(totalsDict.values())
      digit = np.argmax(totals)

      if int(digit) != int(imgValues[elem]):
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

  # ================================ Internal =================================
  def pruneDigitDict(self, printOut=False):
    for key in self.digitDict.keys():
      if printOut:
        print(key,":",len(self.digitDict[key]))
      
      # 6/5/23 DH: Now limit size to 10 elements
      self.digitDict[key] = self.digitDict[key][:10]

  def printDigitArrayValues(self, testImg):
    # Get the image for dictionary key '0'
    #testImg = self.digitDict[0][0]

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

  def saveDigitArrayValues(self, testImg,fileNum=1):
    filename = self.digitFilename + str(fileNum)
    with open(filename, 'w') as fp:

      fp.write("Test img:" + str(type(testImg)) + " " + str(testImg.shape) + " " +
               str(type(testImg[0][0])) + "\n\n")

      # ----------------- Write selected region --------------------
      
      xOffset = 15
      y = 15
      fp.write(str(xOffset) + "," + str(y) + " of " + str(testImg.shape) + "\n")

      for xIdx in range(7):
        xIdx += xOffset

        # 6/5/23 DH: When remove "Convert the sample data from integers to floating-point numbers"
        #            "{:.2f}" becomes "{:3}"

        # This a row printout (despite looking like a column in code)
        fp.write("{:4}".format(testImg[xIdx][y]) +
              "{:4}".format(testImg[xIdx][y+1]) +
              "{:4}".format(testImg[xIdx][y+2]) +
              "{:4}".format(testImg[xIdx][y+3]) +
              "{:4}".format(testImg[xIdx][y+4]) +
              "{:4}".format(testImg[xIdx][y+5]) +
              "{:4}".format(testImg[xIdx][y+6]) +
              "{:4}".format(testImg[xIdx][y+7]) + "\n")
        
      fp.write("\n")
      # ------------------------------------------------------------

      # --------------------- Write whole image --------------------
      xSize, ySize = testImg.shape

      fp.write("\n")
      for x in range(xSize):
        for y in range(ySize):
          fp.write("{:4}".format(testImg[x][y]))
        fp.write("\n")
      # ------------------------------------------------------------

  def getImgCheckTotals(self, img):
    #totals = []
    totalsDict = {}

    # Convert image from float to integer array
    #img = img * 255
    #img = np.asarray(img, dtype = 'uint8')

    for key in self.digitDict.keys():
  
      # First (and only) element of list for each digit 'key'
      testImg = self.digitDict[key][0]

      self.displayImg(img)
      #self.displayImg(testImg)

      # Convert image from float to integer array
      #testImg = testImg * 255
      #testImg = np.asarray(testImg, dtype = 'uint8')

      #print("testImg:",testImg.shape)
      #print("img:",img.shape)

      bitwiseAndRes = numpy.bitwise_and(testImg, img)
      imgX, imgY = bitwiseAndRes.shape
      #print("\n",key,"- Shape:",imgX, imgY)

      iTotal = 0
      iPixValChg = 0
      for x in range(imgX):
        for y in range(imgY):
          #iTotal += bitwiseAndRes[x][y]
          iTotal += 1

          # 9/5/23 DH: appears to be a "hooked" test function that selectively alters the image 
          #            (giving image a mottled effect)
          if bitwiseAndRes[x][y] > 0:
            """
            print("X:",x,"Y:",y,"=",bitwiseAndRes[x][y])
            print("testImg:",testImg[x][y],"img:",img[x][y])
            print("testImg & img:",testImg[x][y] & img[x][y])

            print("{:>10}".format(bin(testImg[x][y])))
            print("{:>10}".format(bin(img[x][y])))
            print("{:>10}".format(bin(bitwiseAndRes[x][y])))
            print("{:>10}".format(bin(testImg[x][y] & img[x][y])))
            print()
            """

            # 9/5/23 DH: Make whole image binary (rather than grey_scale)
            pixVal = int(key+1) * 100
            img[x][y] = pixVal
            iPixValChg += 1

      """
      print("********************")
      print(key,"total:", iTotal)
      print("********************")
      """
      if key < 3:
        print("Key:",key,"PixVal:",pixVal,"iPixValChg:",iPixValChg)
        # Print out binary image to compare with grey_scale image at start of function
        self.printDigitArrayValues(img)
        self.saveDigitArrayValues(img,key)
        self.displayImg(img,timeout=3)
      else:
        sys.exit()

      #totals.append(iTotal)
      totalsDict[key] = iTotal
    # END: ------------ 'for key in self.digitDict.keys()' --------------

    return totalsDict
  
  # --------------------------------------------------------------------------------------------

