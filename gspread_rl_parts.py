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
