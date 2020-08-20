# to ease reading functions and static vars
from sharedGUI import *
# to read dynamic shared vars
import sharedGUI
import os
import openpyxl
import numpy as np
import math
import ImportLib
def showStatScreen():
    statWindow = Toplevel()
    titleText = "إجمالي نتائج " + str(sharedGUI.TEAM_NAME)
    statWindow.title(titleText)
    statWindow.geometry("1100x618")
    statWindow.resizable(width=False,height=False)
    #back ground
    topCanv1 = Canvas(statWindow, width=1100, height=618)
    topCanv1.place(x=0, y=0)
    img1 = PhotoImage(file="images\\sat.gif")
    topCanv1.create_image(0, 0, anchor=NW, image=img1)
    res = getCalculations()
    ## greater than 2/3 Excellent
    ex = Label(statWindow, text=res[0], font = helv36, bg="#FFF")
    ex.place(x=880, y=135)
    ## greater than 1/2 Good
    good = Label(statWindow, text=res[1], font = helv36, bg="#FFF")
    good.place(x=880, y=210)
    ## greater than 1/6 Accepted
    accepted = Label(statWindow, text=res[2], font = helv36, bg="#FFF")
    accepted.place(x=880, y=286)
    ## less than 1/6 Failed
    failed = Label(statWindow, text=res[3], font = helv36, bg="#FFF")
    failed.place(x=880, y=368)
    ## total grade
    failed = Label(statWindow, text=res[4], font = helv36, bg="#FFF")
    failed.place(x=740, y=445)
    ## total shooters
    tshooters = Label(statWindow, text=res[5], font = helv36, bg="#FFF")
    tshooters.place(x=225, y=44)
    ## total correct bullets
    tshots = Label(statWindow, text=res[6], font = helv36, bg="#FFF")
    tshots.place(x=225, y=112)
    # TODO convert to subprocess
    os.system('"'+ImportLib.get_configuration("PATHS", "OutputExcell")+'"')
    statWindow.mainloop()

def getCalculations():
    wb = openpyxl.load_workbook(ImportLib.get_configuration("PATHS", "OutputExcell"))
    ws = wb.active
    bulletsList = [ t.value for t in ws['E'] ][1:]
    bulletsArray = np.array(bulletsList)
    result = []
    ratio1 = int(sharedGUI.NO_OF_BULLETS) * 2/3
    ratio2 = int(sharedGUI.NO_OF_BULLETS) * 3/6
    ratio3 = int(sharedGUI.NO_OF_BULLETS) * 1/6
    ## greater than 2/3 Excellent
    result.append( len(bulletsArray[bulletsArray >= ratio1]) )
    bulletsArray = bulletsArray[bulletsArray < ratio1]
    ## greater than 1/2 Good
    result.append( len(bulletsArray[bulletsArray >= ratio2]) )
    bulletsArray = bulletsArray[bulletsArray < ratio2]
    ## greater than 1/6 Accepted
    result.append( len(bulletsArray[bulletsArray >= ratio3]) )
    bulletsArray = bulletsArray[bulletsArray < ratio3]
    ## less than 1/6 Failed
    result.append( len(  bulletsArray)  )
    ## total grade percentage
    grade =  round(  sum(bulletsList) * 100 / (int(sharedGUI.NO_OF_BULLETS)*len(bulletsList)) ,2)  
    result.append(str(grade) + "%")
    ## total shooters
    result.append( len(bulletsList) )
    ## total correct bullets
    result.append( sum(bulletsList) )
    ## 
    print (bulletsList)
    print (result)
    return result