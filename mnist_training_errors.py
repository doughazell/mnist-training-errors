# 30/4/23 DH: Refactor of TFConfig class
from tf_config_train import *
from tf_config_rl import *

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

class MNISTerrors(object):

  def __init__(self) -> None:
    
    self.tfConfigTrain = TFConfigTrain()
    self.tfConfigRL = TFConfigRL(self.tfConfigTrain)

# 30/3/23 DH:
if __name__ == '__main__':
  mnistErrors = MNISTerrors()

  # 24/4/23 DH:
  x_trainSet = x_train[:700]
  y_trainSet = y_train[:700]
  #x_trainSet = x_train[:2000]
  #y_trainSet = y_train[:2000]

  # 1/4/23 DH: List of dicts for DNN params
  paramDictList = [
    #{'dense1': 784, 'dropout1': None, 'trainingNum': x_train.shape[0], 'epochs': 5, 'runs': 1, 'reruns': 1 },
    
    {'dense1': 20, 'dropout1': None, 'epochs': 1, 'x_trainSet': x_trainSet, 'y_trainSet': y_trainSet,
     'trainingNum': x_trainSet.shape[0], 'runs': 1, 'reruns': 1 },
    ]
  
  # 29/4/23 DH: Need cmd line arg:
  #  train
  #  rl

  #mnistErrors.tfConfigTrain.batchRunAshore(paramDictList)

  # 24/4/23 DH:
  mnistErrors.tfConfigRL.rlRun(paramDictList)