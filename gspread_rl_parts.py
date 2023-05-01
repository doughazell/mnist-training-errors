# 30/4/23 DH: Refactor GSpreadErrors class
from gspread_errors import *

class GSpreadRLparts(GSpreadErrors):
  def __init__(self,spreadsheet,sheet) -> None:
    
    # Get access to parent attributes via 'super()'
    super().__init__(spreadsheet,sheet)
  
  # 29/4/23 DH: Initially forgot to add 'self' arg (which is NOT A KEYWORD JUST CONVENTION) that caused:
  #
  # "TypeError: addRowRLparts() takes 9 positional arguments but 10 were given" with no arg ID's
  # "TypeError: addRowRLparts() got multiple values for argument 'entry_date'" with just args
  #
  def addRowRLparts(self, sheet, entry_date, test_num, part_num, count, start, end, lowest, highest):
    try:
      newrow = []
    
      newrow.append(entry_date)
      newrow.append(test_num)
      newrow.append(part_num)
      newrow.append(count)
      newrow.append(start)
      newrow.append(end)
      newrow.append(lowest)
      newrow.append(highest)

      sheet.append_row(newrow, table_range='A:H')
    except Exception as error:
      print("Error with append_row():",error)

  # 1/5/23 DH: Copied (+ renamed) from 'GSpreadErrors' (the parent class) to create a Join table printout
  def getGSheetsDataRLparts(self, sheet, testnum, list_of_dicts=None):
    try:
      if list_of_dicts == None:
        list_of_dicts = sheet.get_all_records(head=1)
      
      record_num = len(list_of_dicts)

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

      # 1/5/23 DH:
      headings = True
      tailspace = False
      indentStr = "    "

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
        
        if values['Test number'] == testnum:
          tailspace = True

          if headings:
            print(indentStr + keyStr)
            print(indentStr + keyStrUnderline)
            headings = False

          print(indentStr + recordStr)
        # END: "if values['Test number'] == testnum"

      # END: "for...in range(0,record_num)"

      if tailspace:
        print()
      
      return list_of_dicts

    except:
      # 29/5/18 DH: Debug only
      raise
