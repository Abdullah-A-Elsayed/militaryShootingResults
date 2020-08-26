# to ease reading functions and static vars
from sharedGUI import *
# to read dynamic shared vars
import sharedGUI
from rowResultScreen import lineResults
from engineApi import *
import openpyxl 
from pathlib import Path
import os
import math
#sys.path.insert(1, "C:\\Users\\Abdallah Reda\\Downloads\\CVC-19-Documnet-Wallet-\\BackEnd\\visionapp\\Natinal_ID\\src\\backend")
import ImportLib
import statScreen
engineAPI_wrapper = None
def shooting_process(lineNo):
    shootersNo = sharedGUI.NO_OF_TARGETS
    ## test shared vars
    print("shooting type", sharedGUI.SHOOTING_TYPE)
    print("No. shooters", sharedGUI.NO_OF_TARGETS)
    print("No. bullets", sharedGUI.NO_OF_BULLETS)
    print("Team name", sharedGUI.TEAM_NAME)
    ##
    global root
    global engineAPI_wrapper
    if(lineNo == 1) :
        create_total_results_file()
        engineAPI_wrapper = wrapper(sharedGUI.TEAM_NAME, sharedGUI.NO_OF_TARGETS, sharedGUI.SHOOTING_TYPE)
    canvas=""
    #back ground
    canvas = Canvas(root, width=1366, height=768)
    canvas.place(x=0, y=0)
    img = PhotoImage(file=ImportLib.rel_to_abs("images\\main.gif",__file__))
    canvas.create_image(0, 0, anchor=NW, image=img)
    # line no. label
    titleText = "رماية الصف رقم " + str(lineNo) + ' / ' + str( math.ceil(  (len(SHOOTERS_INITIAL_DATA) - 1) / shootersNo)  )
    titleText += '   -   ' + str(sharedGUI.TEAM_NAME)
    lineNoLabel = Label(root, font=S_helva, bg="#ccc", text=titleText)
    lineNoLabel.place(x=570, y=200)

    # start shooting
    # photo= PhotoImage(file=ImportLib.rel_to_abs('images\\sahm.png')
    # image=photo
    startBtn= Button(root,  font=helv36, bg="#ccc",text="لقطة ما قبل الضرب",  command= lambda: startClicked() ) 
    startBtn.place(x=650, y=300)


    # end shooting
    # image=photo
    endBtn= Button(root,  font=helv36, bg="#ccc",text="لقطة ما بعد الضرب",  state=DISABLED, command= lambda: endClicked() ) 
    endBtn.place(x=650, y=400)

    # Next
    photo= PhotoImage(file=ImportLib.rel_to_abs('images\\next.png',__file__))
    nextBtn= Button(root,  image=photo, font=helv36, bg="#ccc",  state=DISABLED, command= lambda: nextClicked() ) 
    nextBtn.place(x=670, y=485)

    # All Results
    allResBTN= Button(root,  text="النتيجة الكلية للوحدة", font=helv36, bg="#ccc", state=DISABLED, command= lambda: allResClicked(), fg='#f00' ) 
    allResBTN.place(x=645, y=600)

    #Restart program
    photoRestart= PhotoImage(file=ImportLib.rel_to_abs('images\\restart3.png',__file__))
    restartBtn= Button(root, image=photoRestart,   bg="#ccc",
            state=NORMAL, command= lambda: restartClicked(),
            ) 
    restartBtn.place(x=1115, y=647)

    def restartClicked():
        global root
        clearChildren(root)
        import mainScreen
        mainScreen.show_main_widget()
    
    def nextClicked () :
        print("next line") 
        clearChildren(root)
        shooting_process(lineNo + 1)

    def startClicked () : 
        startBtn.config(state =  DISABLED)
        engineAPI_wrapper.startRowAction()
        endBtn.config(state =  NORMAL)

    def endClicked () :
        endBtn.config(state =  DISABLED)
        API_RES = engineAPI_wrapper.endRowAction()
        bulletsNoRes =API_RES[0]
        imagesRes =API_RES[1]
        nextBtn.config(state =  NORMAL)
        print("Showing Modal windowww") 
        ##
        check_if_all_shooters_done()
        lineResults (lineNo, bulletsNoRes, imagesRes)
        ##
    def check_if_all_shooters_done():
        global SHOOTERS_INITIAL_DATA
        if (lineNo*shootersNo)>=len(SHOOTERS_INITIAL_DATA) - 1:
            nextBtn.config(state =  DISABLED)
            allResBTN.config(state =  NORMAL)
            return True
        return False

    def allResClicked () : 
        # os.system("& start results.xlsx")
        statScreen.showStatScreen()
    # if(lineNo == 1):
    #     check_if_all_shooters_done()
    root.mainloop()

def create_total_results_file():
    global SHOOTERS_INITIAL_DATA
    global RESULT_HEADERS
    headers =  SHOOTERS_INITIAL_DATA[0] + RESULT_HEADERS
    wb = openpyxl.Workbook()
    sheet = wb.active 
    sheet.title = 'sheet1'
    sheet.append(headers)
    wb.save(ImportLib.rel_to_abs(ImportLib.get_configuration("PATHS", "OutputExcell"),__file__))



