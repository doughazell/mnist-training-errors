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

