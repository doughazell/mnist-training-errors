# -*- coding: utf-8 -*-
"""mnist-training-errors.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-71Qerfz5gfCn53TYLsddTp5sprd6yZS

# MNIST training errors

Based on https://colab.research.google.com/drive/1i_UshCAHw1gj8lJOqZN6pelwWXmcJhbB
"""

"""> Python array slice notation:
```
a[start:stop] = items start through stop-1
a[start:]     = items start through the rest of the array
a[:stop]      = items from the beginning through stop-1
a[:]          = a copy of the whole array
```
"""

import matplotlib.pyplot as plt
import tensorflow as tf

# 25/1/23 DH: Refactor of model creation + DB model access
from models import *

# Python 3 added 'importlib'
#import importlib
#mymethod = getattr(importlib.import_module("models"), "mymethod")

#gotoGSheetTesting()

print("-------------------------------------")
print("TensorFlow version:", tf.__version__)
print("-------------------------------------")

"""## Train and evaluate your model"""
# 28/3/23 DH: 784->10 so {700,600,500,400,300,200,128,100}
dense1 = 700
# 28/3/23 DH: {0.9,...,0.2,0.1,0}
dropout1 = 0
x_trainSet = x_train
y_trainSet = y_train
model = createModel(dense1=dense1, dropout1=dropout1, x_trainSet=x_trainSet, y_trainSet=y_trainSet)

print("\n--- model.evaluate() ---")
print("Using x_train + y_train (%i): "%(x_train.shape[0]))
model.evaluate(x_train,  y_train, verbose=2)
print("***")
print("Using x_test + y_test (%i): "%(x_test.shape[0]))
model.evaluate(x_test,  y_test, verbose=2)

# ----------------------------------------------------------------------
def displayImg(elem):
  
  #print("Dataset Element: ",elem," Expected: ",y_test[elem])

  # 26/1/23 DH: Also need predicted (ie highest softmax) to compare with image
  # (...why can't 'expected/definitive' be compared with 'predicted/softmax', so obviate image display...??? )

  # https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html
  plt.imshow(x_test[elem], cmap='gray_r')
  #plt.draw_if_interactive()
  #self.flush_events()
  
  # 22/1/23 DH: Calling function without '()' does not return an error but does NOT execute it (like function pointer)
  plt.draw()
  plt.waitforbuttonpress(timeout=1)

# ----------------------------------------------------------------------

# 16/1/23 DH: Provide the highest value from each element of the array which the AI determined value of 'x_test'
#             to be compared (by human or ensemble computer vision) against the matching 'y_test'

#print("\n--- probability_model() ---")
probability_model = getProbabilityModel(model)
#getPredicted(0, probability_model)
#print("---------------------------\n")

plt.show()

# 16/1/23 DH: 'x_test' is a 3-D array of 10,000 28*28 images
#print(x_test.shape)
imgNum = x_test.shape[0]
#print("Images: ",imgNum)

#with open("visualization/test_embedding_vectors.tsv", "w") as f:
#  f.write("\n".join (["\t".join(testset_embedding) for testset_embedding in testset_embeddings]) )
#f.close()
f = open("predicted-errors.txt", "w")

softmax2DList = probability_model(x_test).numpy()
softmaxList = softmax2DList[0]
print("\nSoftmax list for element 0 of 'x_test': ",softmaxList )

iCnt = 0
errorNum = 0

print("\nLooping through",imgNum,"images from x_test")
for elem in range(imgNum):
#for elem in range(10):

  #if elem % 7000 == 0:
  #  displayImg(elem)

  # 29/3/23 DH: MAHOOSIVELY sped up checking images by calling 'probability_model(x_test).numpy()' once (above)
  #predictedVal = getPredicted(elem, probability_model)
  predictedVal = np.argmax(softmax2DList[elem])
  if y_test[elem] != predictedVal:
    f.write("Dataset Element: "+ str(elem) + " Expected: "+ str(y_test[elem]) + " Predicted: " + str(predictedVal) + "\n")
    #displayImg(elem)
    iCnt += 1

  if errorNum != iCnt and iCnt % 100 == 0:
    print(iCnt, "errors at element",elem)
    errorNum = iCnt

  lastNum = elem

print("-----------")
print("Last element: ",lastNum)
print("Total errors: ",iCnt)

f.write("-----------\n")
f.write("Last element: " + str(lastNum) + "\n")
f.write("Total errors: " + str(iCnt) + "\n")
f.close()

#displayImg(0)
#displayImg(1)

# ===========================================================================
# 27/3/23 DH: Now add the results of the errors to gsheet
sheet = getGSheet()
updateSheet(sheet,2,9,"ooh yea...")
#addRow(sheet, 128, 0.2, 60000, 10000, 716)
addRow(sheet, dense=dense1, dropout=dropout1, training_num=x_trainSet.shape[0], test_num=imgNum, errors=iCnt)
getGSheetsData(sheet)


