# 30/3/23 DH: Refactor of DNN model creation

import tensorflow as tf
import numpy as np

class TFModel(object):

  def __init__(self) -> None:
    print("-------------------------------------")
    print("TensorFlow version:", tf.__version__)
    print("-------------------------------------")

  # 27/3/23 DH:
  def createModel(self, dense1, dropout1, x_trainSet, y_trainSet):
    # 15/3/23 DH: https://www.kirenz.com/post/2022-06-17-introduction-to-tensorflow-and-the-keras-functional-api/
    #   "The Keras sequential model is easy to use, but its applicability is extremely limited: 
    #    it can only express models with a single input and a single output, 
    #    applying one layer after the other in a sequential fashion.

    #    In practice, it’s pretty common to encounter models with multiple inputs (say, an image and its metadata),
    #    multiple outputs (different things you want to predict about the data), or a nonlinear topology. 
    #    In such cases, you’d build your model using the Functional API."
    #
    # ...well it seems that the Sequential() constructor has this built in now...
    #    https://www.tensorflow.org/guide/keras/functional
    #
    # 29/3/23 DH: The google suggested model uses a Fully Connected DNN (not a Convolution DNN)
    #             https://medium.com/swlh/fully-connected-vs-convolutional-neural-networks-813ca7bc6ee5
    #             
    #             "The "full connectivity" of these networks make them prone to overfitting data."
    #             https://en.wikipedia.org/wiki/Convolutional_neural_network
    # ...hence the 'dropout' (which in my tests made things worse for black lines on white background...!)
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),

      #tf.keras.layers.Dense(128, activation='relu'),      
      tf.keras.layers.Dense(dense1, activation='relu'),

      # 18/3/23 DH: https://keras.io/api/layers/regularization_layers/dropout/
      #   "Dropout layer randomly sets input units to 0 with freq of rate at each step during training time, 
      #    which helps PREVENT OVERFITTING. (ie fuzzy logic) 
      #    Inputs not set to 0 are SCALED UP by 1/(1 - rate) such that SUM over all inputs is UNCHANGED"

      #tf.keras.layers.Dropout(0.2),      
      #tf.keras.layers.Dropout(dropout1),

      # 16/3/23 DH: Condense to decimal digit bucket set
      tf.keras.layers.Dense(10)
    ])
      
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    
    print("\n--- model.fit() ---")
    print("Using x_train + y_train (%i): "%(x_trainSet.shape[0]))
    #model.fit(x_train, y_train, epochs=5)
    model.fit(x_trainSet, y_trainSet, epochs=5)

    #print("Using x_test + y_test (%i): "%(x_test.shape[0]))
    #model.fit(x_test, y_test, epochs=1)

    self.model = model
    return model

  def createSavedModel(self):
    try:
      model = tf.keras.models.load_model("mnist_training_errors")
    except OSError as e:
      # model doesn't exist, build it...

      # 28/3/23 DH:
      model = self.createModel(dense1=128, dropout1=0.2, x_trainSet=self.x_train, y_trainSet=self.y_train)

      # 23/1/23 DH: https://www.tensorflow.org/guide/keras/save_and_serialize
      model.save("mnist_training_errors")

      # 15/3/23 DH:
      tf.keras.utils.plot_model(model,"trg_error-model.png",show_shapes=True)

    return model

  def getProbabilityModel(self, model):
    # 28/1/23 DH: 'tf.keras.Sequential()' return type:  <class 'tensorflow.python.framework.ops.EagerTensor'>
    #
    #             https://www.tensorflow.org/api_docs/python/tf/Tensor :
    # "TensorFlow supports eager execution and graph execution. 
    #  In eager execution, operations are evaluated immediately. In graph execution, a computational graph is constructed for later evaluation.
    #  TensorFlow defaults to eager execution. Note that during eager execution, you may discover your Tensors are actually of type EagerTensor."
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])
    return probability_model

