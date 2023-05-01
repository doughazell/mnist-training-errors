# 30/4/23 DH: Refactor GSpreadErrors class
from gspread_errors import *

class GSpreadRL(GSpreadErrors):
  def __init__(self,spreadsheet,sheet) -> None:
    
    # Get access to parent attributes via 'super()'
    super().__init__(spreadsheet,sheet)
  
  # 29/4/23 DH: https://buildmedia.readthedocs.org/media/pdf/gspread/latest/gspread.pdf
  def getRLTestnum(self, sheet):
    
    #rowCnt = sheet.row_count
    rowCnt = len(sheet.col_values(1))

    row = rowCnt
    col = 2
    valueList = sheet.range(row,col)
    testnumLast = valueList[0].value

    testnum = "XXX"
    if testnumLast.isdigit():
      testnum = int(testnumLast) + 1
    else:
      testnum = 1

    return testnum

  # 29/4/23 DH:
  def addRowRL(self, sheet, dense, dropout, training_num, retrain_num, run_parts, 
               accuracy_start, accuracy_end, lowest_accuracy):
    try:
      newrow = []
      
      # 25/3/23 DH: Date getting added with prepended ' so not recognised as date by gsheet...ffs...!!!
      #             (an opportunity to "sail the luff" rather than "beat to wind")
      self.dateOfEntry = date.today().strftime('%d%b%y')
      print(self.dateOfEntry)
      newrow.append(self.dateOfEntry)

      self.testnum = self.getRLTestnum(sheet)
      newrow.append(self.testnum)
      newrow.append(dense)
      newrow.append(dropout)
      newrow.append(training_num)
      newrow.append(retrain_num)
      newrow.append(run_parts)
      newrow.append(accuracy_start)
      newrow.append(accuracy_end)
      newrow.append(lowest_accuracy)

      sheet.append_row(newrow, table_range='A:J')
    except Exception as error:
      print("Error with append_row():",error)

  # 1/5/23 DH: Copied (+ renamed) from 'GSpreadErrors' (the parent class) to create a Join table printout
  def getGSheetsDataRL(self, sheet, gspreadRLparts):
    try:
      list_of_dicts = sheet.get_all_records(head=1)
      record_num = len(list_of_dicts)

      # 1/5/23 DH: We only want to access gsheet data ONCE FOR EACH SHEET 
      #            (despite looping through 'mnist-rl-parts' for EACH ROW of 'mnist-rl')
      list_of_parts = None

      keyStr = ''
      keyStrUnderline = ''
      keyList = list_of_dicts[0].keys()

      # 1/5/23 DH: Desired Join table output:
      """
      Date     Test number, Dense, Dropout, Training number, Retrain number, Run parts, ...
      ----     -----------  -----  -------  ---------------  --------------  ---------
      01May23, 9,           20,    ,        700,             55,             2,

          Date     Test number, Part number, Count, Start, End, Lowest, Highest, 
          ----     -----------  -----------  -----  -----  ---  ------  ------- 
          01May23, 9,           1,           50,    0.42,  0.46,0.39,   0.46,    
          01May23, 9,           2,           5,     0.47,  0.52,0.47,   0.52,
      """

      # Headings
      # --------
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

      # --- Row values ---
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
        
        # 1/5/23 DH: Print each test num row and then call for output from 'mnist-rl-parts'
        print(recordStr)

        testnum = values['Test number']
        list_of_parts = gspreadRLparts.getGSheetsDataRLparts(gspreadRLparts.sheet, testnum, list_of_parts)

      # END: "for...in range(0,record_num)"

      print()

    except:
      # 29/5/18 DH: Debug only
      raise


