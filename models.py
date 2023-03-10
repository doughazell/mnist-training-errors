# 25/1/23 DH: Refactor of NN model creation + DB model

import tensorflow as tf
import numpy as np

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
def createModel():
  try:
    model = tf.keras.models.load_model("mnist_training_errors")
  except OSError as e:
    # model doesn't exist, build it.
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10)
    ])
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    #print("x_train[:1]: ",x_train[:1].shape)
    #predictions = model(x_train[:1]).numpy()
    #tf.nn.softmax(predictions).numpy()
    #loss_fn(y_train[:1], predictions).numpy()

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    print("\n--- model.fit() ---")
    #25/1/23 DH: model.fit(x_train[:30000], y_train[:30000], epochs=1)
    
    #print("Using x_train + y_train (%i): "%(x_train.shape[0]))
    #model.fit(x_train, y_train, epochs=1)
    
    print("Using x_test + y_test (%i): "%(x_test.shape[0]))
    model.fit(x_test, y_test, epochs=1)

    # 23/1/23 DH: Can the model be saved after training to prevent the need for retraining...???
    # ...yip...https://www.tensorflow.org/guide/keras/save_and_serialize
    model.save("mnist_training_errors")

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
def getPredicted(elem, probability_model):
  
  # 26/1/23 DH: Print index of highest prob ie predicted number
  mnistIndex = elem
  
  #softmaxList = probability_model(x_test[:mnistIndex+1]).numpy()[0]
  # 28/1/23 DH:
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
from oauth2client.service_account import ServiceAccountCredentials

def getHeadings(sheet):
    '''
    print self.list_of_dicts

    print '-----------------------------'
    keyList = sorted(self.list_of_dicts[0].keys())
    print keyList
    print keyList[0]
    print '-----------------------------'
    '''

    print ('REQUEST TO DB for row 1 (ie headings)')
    headings = filter(None, sheet.row_values(1))

    #lastCol = len(headings)
    #orderCell = [1,lastCol]

    # 28/6/18 DH: Col order is the last heading
    # 26/1/23 DH: https://diveintopython3.net/porting-code-to-python-3-with-2to3.html#filter
    headingsList = list(headings) 
    order = headingsList.pop()
    print ('Order: ', order)
    txt1_text = order

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


def getGSheetsData():
    try:
        # 2/11/17 DH:
        # use creds to create a client to interact with the Google Drive API
        # 9/7/18 DH: Upgrade to gspread >2.0.0
        #scope = ['https://spreadsheets.google.com/feeds']
        scope = ['https://spreadsheets.google.com/feeds',
                    'https://www.googleapis.com/auth/drive']

        creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
        # 11/7/18 DH: Attempting to fix auth error with Google Sheets API in upgrade to v3.0.1
        #creds = ServiceAccountCredentials.from_json_keyfile_name('python-sheets.json', scope)
        client = gspread.authorize(creds)

        print('Opening \'Addresses:Personal\'...')
        sheet = client.open("Addresses").worksheet("Personal")

        print('REQUEST TO DB for all records')
        list_of_dicts = sheet.get_all_records(head=1)

        record_num = len(list_of_dicts)

        lbl1_text = str(record_num) + ' records'

        hdsIndexed = getHeadings(sheet)

        #cols = main.txt1.text.split(",")
        cols = "3,4,5,6,7".split(",")

        # === Rows ===
        for idx in range(0,record_num):
            lbl_text = ''

            print('REQUEST TO DB for row ',str(idx+2))
            #values = sheet.row_values(idx+2)

            values = list_of_dicts[idx]

            if values:
                # ||| Cols |||

                # 22/3/18 DH: Selected cols added as specified in TextInput 'txt1'
                colsIndexed = dict(zip(range(1,8), values))

                #print '-----------------------------'
                #print values
                #print self.hdsIndexed

                for col in cols:
                    #ROW: values = self.list_of_dicts[idx]
                    #COL HEADINGS: self.hdsIndexed
                    #printStr = str(col) + ' = ' + hdsIndexed.get(int(col)) + ' = ' + values.get(hdsIndexed.get(int(col)))
                    #print(printStr)
                    #print '-----------------------------'

                    #lbl_text += colsIndexed.get(int(col)) + '\n'
                    lbl_text += str(values.get( hdsIndexed.get(int(col)) )) + '\n'

                # Add record to carousel
                print ("Adding:")
                print (lbl_text)
                

    except:
        # 29/5/18 DH: Debug only
        #print values
        raise

