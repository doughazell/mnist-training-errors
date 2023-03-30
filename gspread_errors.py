import gspread

# 20/3/23 DH: https://oauth2client.readthedocs.io/en/latest/_modules/oauth2client/service_account.html#ServiceAccountCredentials
# 20/3/23 DH: https://google-auth.readthedocs.io/en/latest/oauth2client-deprecation.html
from oauth2client.service_account import ServiceAccountCredentials

# 25/3/23 DH:
from datetime import date

class GSpreadErrors(object):

  #def __new__(self) -> gspread.Spreadsheet:
  #  return sheet
  
  def __init__(self) -> None:
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
      self.sheet = client.open("Addresses").worksheet("mnist-errors")
    
    except:
      raise
  
  def getGSheetsData(self, sheet):
    try:
      print('\nREQUEST TO DB for all records')
      list_of_dicts = sheet.get_all_records(head=1)
      record_num = len(list_of_dicts)

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

      # END: "for...in range(0,record_num)"

      print()

    except:
      # 29/5/18 DH: Debug only
      raise

  # 19/3/23 DH: gspread.exceptions.APIError: {'code': 400, 
  # 'message': 'Range (Personal!H5) exceeds grid limits. Max rows: 4, max columns: 26', 
  # 'status': 'INVALID_ARGUMENT'}
  def exceedsGridLimits(self, sheet, row, col, respStr):
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

  # 24/3/23 DH: Needs a 'pytest' string parsing auto test
  def checkEmpty(self, sheet, row, col, text, splitStr=None):
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
  def updateSheet(self, sheet, row, col, text):
    try:
      text = self.checkEmpty(sheet, row, col, text, splitStr="...")
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
        self.exceedsGridLimits(sheet,row,col,respStr)
        self.updateSheet(sheet,row,col,text)
      else:
        print(respStr)

  # 25/3/23 DH:
  def addRow(self, sheet, dense, dropout, training_num, test_num, errors):
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

