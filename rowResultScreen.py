from sharedGUI import *
import sharedGUI
import os
import math
from PIL import ImageTk, Image
#sys.path.insert(1, "C:\\Users\\Abdallah Reda\\Downloads\\CVC-19-Documnet-Wallet-\\BackEnd\\visionapp\\Natinal_ID\\src\\backend")
import ImportLib
def lineResults (lineNo, buletsNoResult, shooterImageResult):
    SHOOTERS_NO = sharedGUI.NO_OF_TARGETS
    global SHOOTERS_INITIAL_DATA #list of lists
    lineResultWindow = Toplevel()
    titleText = "نتائج رماية الصف رقم " + str(lineNo) + ' / ' + str( math.ceil(  (len(SHOOTERS_INITIAL_DATA) - 1) / SHOOTERS_NO)  )
    titleText += '   -   ' + str(sharedGUI.TEAM_NAME)
    lineResultWindow.title(titleText)
    lineResultWindow.geometry("1100x628")
    lineResultWindow.resizable(width=False,height=False)

    #back ground
    topCanv1 = Canvas(lineResultWindow, width=1100, height=628)
    topCanv1.place(x=0, y=0)
    # img1 = PhotoImage(file=ImportLib.rel_to_abs("images\\file test2, __file__)2.gif")
    img1 = PhotoImage(file=ImportLib.rel_to_abs("images\\blank.gif", __file__))
    topCanv1.create_image(0, 0, anchor=NW, image=img1)
    
    # take data from global SHOOTERS_INITIAL_DATA that exists in sharedGUI
    firstShooterOrder = SHOOTERS_NO*(lineNo-1)+1
    data = getCurrentRowInitialData(SHOOTERS_INITIAL_DATA,firstShooterOrder,SHOOTERS_NO)
    ## appending results to tk
    ### bullets no
    global RESULT_HEADERS
    data[0] = data[0] + RESULT_HEADERS
    for i in range(len(data)-1):
        # append bullets no
        data[i+1] = data[i+1] + [buletsNoResult[i]]
        # append image paths
        data[i+1] = data[i+1] + [shooterImageResult[i]]
        # print (data)
        # TODO APPEND OTHER THINGS
    '''data = [('التقدير', 'النتيجة', 'الوحدة','الاسم','الرتبة',"م"),
        ('ممتاز', 6, 'السرية الثامنة','عبد الله علي السيد','طالب',1), 
        ('ممتاز', 5, 'السرية السابعة','عبد الرحمن أحمد السيد','طالب',2), 
        ('ضعيف', 0, 'السرية الأولى','عبد الله أحمد هشام','طالب',3), 
        ('مقبول', 2, 'السرية الثامنة','حودة أحمد فتيح','طالب',4), 
        ('جيد', 3, 'السرية التاسعة','عبد الله أحمد هشام','طالب',5)] 
    '''
    ### append results to excell sheet
    #outputname = Path('.','results.xlsx')
    outputname = ImportLib.rel_to_abs('results.xlsx', __file__)
    wb = openpyxl.load_workbook(outputname)
    sheet = wb.active 
    sheet.title = 'sheet1'
    for i in range(len(data)):
        # if not heders
        if(i!=0):
            sheet.append(data[i])  
    wb.save(outputname)
    ###
    # add table to gui
    tblFrame = Frame(lineResultWindow, bd=10, relief=RAISED)
    print(data)
    t = Table(tblFrame, data, pushButtonsBack = True) 
    tblFrame.pack(side="left", expand=1)
    lineResultWindow.mainloop()

def getCurrentRowInitialData(data, firstOrder, shootersNo):
    res = []
    res.append(data[0])
    res = res + data[firstOrder:firstOrder+shootersNo]
    return res

def showShooterImage(imgPath):
    print(imgPath)
    # shooterTargetImage = Toplevel()
    # shooterTargetImage.title("صورة الهدف ")
    # shooterTargetImage.geometry("1100x650")
    # shooterTargetImage.resizable(width=True,height=True)
    # #img = PhotoImage(file=imgPath)
    # img = ImageTk.PhotoImage(Image.open(imgPath))
    # c = Canvas(shooterTargetImage, width=1100, height=650)
    # c.create_image(0, 0, anchor=NW, image=img)
    # c.pack()
    # shooterTargetImage.mainloop()
    #print('start "'+imgPath+'"')
    os.system('"'+imgPath+'"')