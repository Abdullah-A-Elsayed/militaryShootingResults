from tkinter import *
import tkinter.font as tkFont
import openpyxl
from pathlib import Path
#sys.path.insert(1, "C:\\Users\\Abdallah Reda\\Downloads\\CVC-19-Documnet-Wallet-\\BackEnd\\visionapp\\Natinal_ID\\src\\backend")
import ImportLib

root = Tk()
root.title("برنامج رصد نتائج الرماية")
root.resizable(width=False,height=False)
dim="1366x768"
root.geometry(dim)
helv36 = tkFont.Font(root, family='Helvetica', size=20, weight='bold')
helv36_sm = tkFont.Font(root, family='Helvetica', size=15, weight='bold')
head1 = tkFont.Font(root, family='Helvetica', size=25, weight='bold')
beIN = tkFont.Font(root, family='Helvetica', size=20, weight='bold')
waitFont = tkFont.Font(root, family='Helvetica', size=50, weight='bold')
FinFont = tkFont.Font(root, family='Helvetica', size=40, weight='bold')
S_helva = tkFont.Font(root, family='Helvetica', size=25, weight='bold')
arNum=['۰','۱','۲','۳','٤','٥','٦','٧','۸','۹']
def trim(myStr):
   newStr = myStr.rstrip()
   return newStr.lstrip() 
def clearChildren(parent):
    for childWidget in parent.winfo_children():
       childWidget.destroy()

   
 ##### table #####
class Table: 
    def __init__(self,parent, data, pushButtonsBack = False): 
        # data is a list of tuples
        _rows = len(data) 
        _columns = len(data[0]) 
       
        # code for creating table 
        import rowResultScreen
        for i in range(_rows): 
            def make_show_image_lambda(path):
                return lambda ev:rowResultScreen.showShooterImage(path)
            for j in range(_columns): 
                _fg='black'
                _fSize=16
                if(i==0):
                    _fg='red'
                    _fSize=20

                #if custom behaviour (res image button)
                if(pushButtonsBack and j== 0 and i!=0):
                    self.e = Button(parent, font=('Arial',20,'bold'), bg="#ccc",text="📷")
                    path = data[i][_columns-1-j]
                    self.e.bind("<Button-1>", make_show_image_lambda(path))
                    self.e.grid(row=i, column=j, pady=1) 

                # normal behaviour
                else:
                    self.e = Label(parent, text= data[i][_columns-1-j], fg=_fg, 
                        font=('Arial',_fSize,'bold'), padx=25,
                        #bd=6, relief=RAISED,
                    )
                    self.e.grid(row=i, column=j, pady=1) 
       

def readFromExcell(directory, filename):
    #xlsx_file = Path(directory, filename)
    #xlsx_file = "C:\\Users\\Abdallah Reda\\Downloads\\CVC-19-Documnet-Wallet-\\BackEnd\\visionapp\\Natinal_ID\\src\\gui\\الضاربون.xlsx"
    xlsx_file = ImportLib.rel_to_abs(filename, __file__)
    wb_obj = openpyxl.load_workbook(xlsx_file)

    # Read the active sheet:
    sheet = wb_obj.active

    namesList=[]
    for row in sheet.iter_rows():
        temp =[]
        for cell in row:
            temp.append(str(cell.value))
        namesList.append(temp)
    return (namesList)
NO_OF_TARGETS = 0
NO_OF_BULLETS = 0
SHOOTING_TYPE = ''
TEAM_NAME = ''
# print("====="+ImportLib.get_configuration("PATHS","InputExcell")+"=====")
SHOOTERS_INITIAL_DATA = readFromExcell('.', ImportLib.get_configuration("PATHS","InputExcell"))
RESULT_HEADERS = ['النتيجة','الصورة',]