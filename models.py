# 25/1/23 DH: Refactor of NN model creation + DB model

import tensorflow as tf
import numpy as np

# 19/3/23 DH:
import sys

# 25/3/23 DH: overridden since still created '__pycache__/models.cpython-39.pyc'
#             'python -B' still caused "'date" in gsheet hence string not date...!
#sys.dont_write_bytecode = True

"""## Load a dataset

Load and prepare the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). Convert the sample data from integers to floating-point numbers 

("The training set contains 60000 examples, and the test set 10000 examples") :
"""

# http://yann.lecun.com/exdb/mnist/
mnist = tf.keras.datasets.mnist

# https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html
# 'x_train.shape[0]' = take first col number of 'nD' array
#print("\nNumber of entries => Inputs: ",x_train.shape[0], "{",x_train.size,"=",x_train.shape,"}, Labels: ",y_train.shape)
#print("x_train type: ",type(x_train))


# ----------------------------------------------------------------------
# 27/3/23 DH:
def createModel(dense1, dropout1, x_trainSet, y_trainSet):
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

    # 15/3/23 DH: Learn about TensorFlow training by finding total error diff of removing following sections
    #             https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
    # 16/3/23 DH: 784->10 so {700,600,500,400,300,200,128,100}
    #tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(dense1, activation='relu'),

    # 18/3/23 DH: https://keras.io/api/layers/regularization_layers/dropout/
    #   "Dropout layer randomly sets input units to 0 with freq of rate at each step during training time, 
    #    which helps PREVENT OVERFITTING. (ie fuzzy logic) 
    #    Inputs not set to 0 are SCALED UP by 1/(1 - rate) such that SUM over all inputs is UNCHANGED"

    # 16/3/23 DH: {1.0,0.9,...,0.2,0.1,0}
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dropout(dropout1),

    # 16/3/23 DH: Condense to decimal digit bucket set
    tf.keras.layers.Dense(10)
  ])
  
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=['accuracy'])

  print("\n--- model.fit() ---")
  print("Using x_train + y_train (%i): "%(x_train.shape[0]))
  #model.fit(x_train, y_train, epochs=5)
  model.fit(x_trainSet, y_trainSet, epochs=5)

  #print("Using x_test + y_test (%i): "%(x_test.shape[0]))
  #model.fit(x_test, y_test, epochs=1)

  return model


def createSavedModel():
  try:
    model = tf.keras.models.load_model("mnist_training_errors")
  except OSError as e:
    # model doesn't exist, build it...

    # 27/3/23 DH:
    #model = createModel()

    # 28/3/23 DH:
    model = createModel(dense1=128, dropout1=0.2, x_trainSet=x_train, y_trainSet=y_train)

    # 23/1/23 DH: https://www.tensorflow.org/guide/keras/save_and_serialize
    model.save("mnist_training_errors")

    # 15/3/23 DH:
    tf.keras.utils.plot_model(model,"trg_error-model.png",show_shapes=True)

  return model

# ----------------------------------------------------------------------

def getProbabilityModel(model):
    #predictionsTest = tf.nn.softmax(x_test[0])

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

# 27/1/23 DH:
# 29/3/23 DH: MAHOOSIVELY sped up checking images by calling 'probability_model(x_test).numpy()' once,
#             rather than entering ALL images ('x_test' input to {DNN + Softmax layer} ) for EACH image.
def getPredicted(elem, probability_model):
  
  # 26/1/23 DH: Print index of highest prob ie predicted number
  mnistIndex = elem
  
  # 28/1/23 DH:
  # 29/3/23 DH: This is what takes the 5-10mins time in checking predicted for 10000 images...!
  softmaxList = probability_model(x_test).numpy()[mnistIndex]

  # 26/1/23 DH: 'numpy.ndarray' object has no attribute 'index'
  # softmaxList.index( max(softmaxList) ) 

  #index_min = np.argmin(values)
  softmaxIndex = np.argmax(softmaxList)
  
  #print("Model output for element: ",mnistIndex," => ", softmaxIndex,": ", max(softmaxList) )

  # Print first 1st element of 10000
  #print("keras.Sequential(0): ", probability_model(x_test[:1]))
  #print("tf.keras.Sequential() return type: ", type(probability_model(x_test[:1])) )

  # Convert 'tf.Tensor([[...]], shape=(28, 28), dtype=float64)' into 'numpy([[...]])'
  #predictions = model(x_train[:1]).numpy()

  return softmaxIndex

# ============================================================================
# 25/1/23 DH: Add google sheets code from 'kivy-gapp' ('table.py', 'main.py')
# ----------------------------------------------------------------------------
import gspread

# 20/3/23 DH: https://oauth2client.readthedocs.io/en/latest/_modules/oauth2client/service_account.html#ServiceAccountCredentials
# 20/3/23 DH: https://google-auth.readthedocs.io/en/latest/oauth2client-deprecation.html
from oauth2client.service_account import ServiceAccountCredentials

# 25/3/23 DH:
from datetime import date

# 18/3/23 DH:
def getGSheet():
  try:
    # 2/11/17 DH: use creds to create a client to interact with the Google Drive API

    # 9/7/18 DH: Upgrade to gspread >2.0.0
    #scope = ['https://spreadsheets.google.com/feeds']
    scope = ['https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive']

    creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
    # 11/7/18 DH: Attempting to fix auth error with Google Sheets API in upgrade to v3.0.1
    #creds = ServiceAccountCredentials.from_json_keyfile_name('python-sheets.json', scope)
    client = gspread.authorize(creds)

    #print('Opening \'Addresses:Personal\'...')
    #sheet = client.open("Addresses").worksheet("Personal")

    # 21/3/23 DH:
    print('Opening \'Addresses:mnist-errors\'...')
    sheet = client.open("Addresses").worksheet("mnist-errors")
    return sheet
  
  except:
    raise

# 26/3/23 DH: 'get_all_records()' "head (int) – Determines which row to use as keys...:
#             (so 'getHeadings()' is a legacy from kivy-gapp development and unnecessary)
def getHeadings(sheet):
  # 28/6/18 DH: Col order is the last heading
  # 26/1/23 DH: https://diveintopython3.net/porting-code-to-python-3-with-2to3.html#filter
  print ('REQUEST TO DB for row 1 (ie headings)')
  headings = filter(None, sheet.row_values(1))
  headingsList = list(headings) 

  lbl2_text = ''
  hdsIndexed = {}
  col = 1
  for heading in headingsList:
      hdsIndexed[col] = heading
      printStr = 'Col ' + str(col) + ' = ' + heading
      print(printStr)

      lbl2_text += str(col) + ' = ' + heading + '\n'
      col += 1
  #print(hdsIndexed)
  return hdsIndexed

# 20/3/23 DH: Needs a refactor from kivy-gapp hack of gsheet app in 2018
# 26/3/23 DH: ...done
def getGSheetsData(sheet):
  try:
    print('\nREQUEST TO DB for all records')
    list_of_dicts = sheet.get_all_records(head=1)
    record_num = len(list_of_dicts)

    # 26/3/23 DH: 'get_all_records()' "head (int) – Determines which row to use as keys...:
    #             (so 'getHeadings()' is a legacy from kivy-gapp development and unnecessary)
    #hdsIndexed = getHeadings(sheet)

    keyStr = ''
    keyStrUnderline = ''
    keyList = list_of_dicts[0].keys()

    for key in keyList:
      if key:
        if "Date" in key:
          keyLenDelta = 5
          keyStr += key + (' ' * keyLenDelta)
          keyStrUnderline += ('-' * len(key) ) + (' ' * keyLenDelta)
        else:
          keyStr += key + ", "
          keyStrUnderline += ('-' * len(key) ) + '  '
    
    print(keyStr)
    print(keyStrUnderline)

    for idx in range(0,record_num):
      values = list_of_dicts[idx]
      #print("Record ",idx+1,":",values)

      recordStr = ''

      for key in keyList:
        if key:
          # 26/3/23 DH: Number of trailing spaces is determined by heading string len, except date
          cellValue = str(values[key])
          cellValueLen = len(cellValue)

          if "Date" in key:
            keyLen = 8
          else:
            keyLen = len(key) + 1

          recordStr += cellValue + "," + (' ' * (keyLen - cellValueLen ) )
      
      print(recordStr)

      # -----------------------------------------------------------------------------------------
      # 26/3/23 DH: Legacy from kivy-gapp where display order was specified by "cols" cell string
      #lbl_text = ''
      #if values:
        #for col in cols:
          #ROW: values = self.list_of_dicts[idx]
          #COL HEADINGS: self.hdsIndexed
       
          #lbl_text += str(values.get( hdsIndexed.get(int(col)) )) + '\n'
      # -----------------------------------------------------------------------------------------
    # END: "for...in range(0,record_num)"

    print()

  except:
    # 29/5/18 DH: Debug only
    raise

# 19/3/23 DH: gspread.exceptions.APIError: {'code': 400, 
# 'message': 'Range (Personal!H5) exceeds grid limits. Max rows: 4, max columns: 26', 
# 'status': 'INVALID_ARGUMENT'}
def exceedsGridLimits(sheet,row,col,respStr):
  respDict = eval(respStr)
  #print(type(respDict))
  print(respDict['message'])

  # 'message': 'Range (Personal!H5) exceeds grid limits. Max rows: 4, max columns: 26'
  respStrParts = respDict['message'].split(".")
  # 'Max rows: 4, max columns: 26'
  respMsgParts = respStrParts[1].lower().split(",")

  msgPartDict = {}
  for msgPart in respMsgParts:
    msgParts = msgPart.split(":")
    k = msgParts[0].strip()
    v = msgParts[1].strip()

    msgPartDict[k] = v

  # 19/3/23 DH: https://docs.gspread.org/en/v3.7.0/api.html#gspread.models.Worksheet.resize
  if row > int(msgPartDict['max rows']):
    print("Need to add row since",row,"is greater than",msgPartDict['max rows'])
    sheet.resize(rows=row)

  if col > int(msgPartDict['max columns']):
    print("Need to add col since",col,"is greater than",msgPartDict['max columns'])
    sheet.resize(cols=col)

# 23/3/23 DH: ...still in energy lockdown like kivy-gapp in 2017...
# 24/3/23 DH: Needs a 'pytest' string parsing auto test
def checkEmpty(sheet, row, col, text, splitStr=None):
  valueList = sheet.range(row,col)
  #print("Value at",row,",",col,":",valueList[0].value)

  if valueList[0].value and text in valueList[0].value:
    
    valueParts = valueList[0].value.split(splitStr)
    
    if not valueParts[1]:
      valueParts[1] = str(2)
    else:
      try:
        valueParts[1] = str(int(valueParts[1]) + 1)

      except (ValueError) as error:
        print(error)
        print("The split string is",splitStr)

    if splitStr:
      newValue = valueParts[0] + splitStr + valueParts[1]
      return newValue

  return text

# 18/3/23 DH:
def updateSheet(sheet, row, col, text):
  try:
    text = checkEmpty(sheet, row, col, text, splitStr="...")
    print("Adding",text,"to",row,",",col)
    sheet.update_cell(row, col, text) 

  except (gspread.exceptions.APIError) as response:
    #print(type(response))
    #print(dir(response))

    # 19/3/23 DH: https://docs.python.org/3/library/exceptions.html#Exception
    #             /Users/doug/.pyenv/versions/3.9.15/lib/python3.9/site-packages/gspread/exceptions.py
    respStr = str(response)

    # gspread.exceptions.APIError: {'code': 400, 
    # 'message': 'Range (Personal!H5) exceeds grid limits. Max rows: 4, max columns: 26', 
    # 'status': 'INVALID_ARGUMENT'}
    if "exceeds grid limits" in respStr:
      exceedsGridLimits(sheet,row,col,respStr)
      updateSheet(sheet,row,col,text)
    else:
      print(respStr)

# 25/3/23 DH:
def addRow(sheet, dense, dropout, training_num, test_num, errors):
  try:
    newrow = []
    
    # 25/3/23 DH: Date getting added with prepended ' so not recognised as date by gsheet...ffs...!!!
    #             (an opportunity to "sail the luff" rather than "beat to wind")
    today = date.today().strftime('%d%b%y')
    print(today)
    newrow.append(today)

    newrow.append(dense)
    newrow.append(dropout)
    newrow.append(training_num)
    newrow.append(test_num)
    newrow.append(errors)

    sheet.append_row(newrow, table_range='A:F')
  except Exception as error:
    print("Error with append_row():",error)

# ========================================== TESTING ============================================
def gotoGSheetTesting():
  try:
    sheet = getGSheet()
    updateSheet(sheet,2,9,"ooh yea...")
    addRow(sheet, 128, 0.2, 60000, 10000, 716)
    getGSheetsData(sheet)

  except:
    raise

  sys.exit()

