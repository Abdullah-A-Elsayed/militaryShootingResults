from sharedGUI import *
from shootingScreen import shooting_process
#sys.path.insert(1, "C:\\Users\\Abdallah Reda\\Downloads\\CVC-19-Documnet-Wallet-\\BackEnd\\visionapp\\Natinal_ID\\src\\backend")
import ImportLib
from Shoot import ShootingStringTypes
def show_main_widget():
    def gunChangedEvent(ev):
        trainBullNoEnt.delete(1.0, END)
        trainRangeEnt.delete(1.0, END)
        sel = optionVar.get()
        if(sel == ShootingStringTypes.AK47):
            trainBullNoEnt.insert(1.0, ImportLib.get_configuration("VALUES","AK47BulletsCount")) 
            trainRangeEnt.insert(1.0, ImportLib.get_configuration("VALUES","AK47TargetsCount")) # Targets  
        
        if(sel == ShootingStringTypes.MORRIS):
            trainBullNoEnt.insert(1.0, ImportLib.get_configuration("VALUES","MORRISBulletsCount")) 
            trainRangeEnt.insert(1.0, ImportLib.get_configuration("VALUES","MORRISTargetsCount")) # Targets  
        
        if(sel == ShootingStringTypes.PISTOL):
            trainBullNoEnt.insert(1.0, ImportLib.get_configuration("VALUES","PISTOLBulletsCount")) 
            trainRangeEnt.insert(1.0, ImportLib.get_configuration("VALUES","PISTOLTargetsCount")) # Targets  
        
    global root
    _trainType= ShootingStringTypes.PISTOL
    _shooters= ImportLib.get_configuration("VALUES","TargetsCount")
    _bulletNo=  ImportLib.get_configuration("VALUES","BulletsCount")
    _team="السرية الثامنة"
    canvas=""
    #back ground
    canvas = Canvas(root, width=1366, height=768)
    canvas.place(x=0, y=0)
    img = PhotoImage(file=ImportLib.rel_to_abs("images\\settings 2.gif",__file__))
    canvas.create_image(0, 0, anchor=NW, image=img)
    #current_element.append(canvas)

    
    # train no-type input
    optionVar = StringVar(root)
    optionVar.set(_trainType)  # default value

    trainTypeEnt= OptionMenu(root, optionVar,
    ShootingStringTypes.AK47 ,
     ShootingStringTypes.MORRIS,
     ShootingStringTypes.PISTOL,
     command = gunChangedEvent,
     )
    trainTypeEnt.config(font=helv36, bg='#ccc', width=10, height="2")
    trainTypeEnt['menu'].config(font=helv36, bg='#d2d2d2')
    trainTypeEnt.place(x=900, y=415)
    #current_element.append(trainTypeEnt)


    # shooters input
    trainRangeEnt = Text(root, font=S_helva, width="6", height="2", bg="#b2b2b2")
    trainRangeEnt.insert(1.0, _shooters)
    trainRangeEnt.place(x=747, y=415)
    #current_element.append(trainRangeEnt)


    # train bullet number input
    trainBullNoEnt = Text(root, font=S_helva, width="6", height="2", bg="#b2b2b2")
    trainBullNoEnt.insert(1.0, _bulletNo)
    trainBullNoEnt.place(x=587, y=415)
    #current_element.append(trainBullNoEnt)

    # Notes input
    teamEnt = Text(root, font=helv36, width="24", height="2", bg="#b2b2b2")
    teamEnt.insert(1.0, _team)
    teamEnt.place(x=185, y=415)

    #start button
    photo= PhotoImage(file=ImportLib.rel_to_abs('images\\sahm.png',__file__))
    # trainRangeEnt.get(1.0,2.0)
    startBtn= Button(root, state=NORMAL,  font=helv36, bg="#ccc", image=photo, command= lambda: config_done_action( optionVar.get(), trainRangeEnt.get(1.0,END), trainBullNoEnt.get(1.0,END), teamEnt.get(1.0,END) ) )
    startBtn.place(x=160, y=560)
    #current_element.append(startBtn)

    root.mainloop()

def config_done_action(trainName, shootersNo, bulletsNo, teamName):
   trainName = trim(trainName)
   shootersNo = trim(shootersNo)
   bulletsNo = trim(bulletsNo)
   teamName = trim(teamName)
   # fill shared variables
   import sharedGUI
   sharedGUI.NO_OF_TARGETS = int(shootersNo)
   sharedGUI.NO_OF_BULLETS = bulletsNo
   sharedGUI.TEAM_NAME = teamName
   sharedGUI.SHOOTING_TYPE = trainName
   # navigation
   global root
   clearChildren(root)
   shooting_process(1)