import matplotlib.pyplot as plt

# 30/3/23 DH: Refactor of model creation + DB model access
from tf_model import *
#from gspread_errors import *

class TFConfig(object):

  def __init__(self, integer=False) -> None:
    # 8/5/23 DH:
    """
    Load and prepare the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). 
    Convert the sample data from integers to floating-point numbers (WHY...???)

    ("The training set contains 60000 examples, and the test set 10000 examples")
    """
    mnist = tf.keras.datasets.mnist
    # https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data()
    (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

    # 6/5/23 DH: WHY...????
    #  '$ tf-test train' appears to be more accurate for floats vs integers 
    #  (5800 vs 7600 for 20Dense-700Train)
    if integer is False:
      self.x_train = self.x_train / 255.0
      self.x_test = self.x_test / 255.0

    self.tfModel = TFModel()

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
    print("Using x_test + y_test (%i): "%(self.x_test.shape[0]))
    evalRes = self.tfModel.model.evaluate(self.x_test,  self.y_test, verbose=2)
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

