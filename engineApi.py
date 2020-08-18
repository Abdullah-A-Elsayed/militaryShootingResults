import sys
import os
from pathlib import Path
import subprocess

# sys.path.insert(1, "C:\\Users\\Abdallah Reda\\Downloads\\CVC-19-Documnet-Wallet-\\BackEnd\\visionapp\\Natinal_ID\\src\\backend")
import Shoot
import sharedGUI
#full_command = 'open "C:\\Users\\Abdallah Reda\\Downloads\\CVC-19-Documnet-Wallet-\\BackEnd\\visionapp\\Natinal_ID\\output\\1_result.jpg"'
#filename = "C:\\Users\\Abdallah Reda\\Downloads\\CVC-19-Documnet-Wallet-\\BackEnd\\visionapp\\Natinal_ID\\output\\1_result.jpg"
#os.system('"start "C:\\Users\\Abdallah Reda\\Downloads\\CVC-19-Documnet-Wallet-\\BackEnd\\visionapp\\Natinal_ID\\output\\1_result.jpg""')
#subprocess.run(['open', filename], check=True)
#p = subprocess.Popen(full_command, stdout=subprocess.PIPE, universal_newlines=True, shell=False)
#(output, err) = p.communicate()
class wrapper:
    def __init__(self, team_name, no_of_targets, shooting_type):
        save_path = os.path.join(os.getcwd(), 'results', team_name)
        save_path += '\\'
        #save_path = save_path.encode('UTF-8')
        print(save_path)
        self.sr = Shoot.ShootingResults(save_path, no_of_targets, shooting_type=wrapper.encode_shooting_type(shooting_type))
    def encode_shooting_type(shooting_type):
        if(shooting_type=="آلي نهاري"):
            return Shoot.ShootingTypes.AK47
        if(shooting_type=="موريس نهاري"):
            return Shoot.ShootingTypes.MORRIS
        if(shooting_type=="طبنجة (۹ ملي)"):
            return Shoot.ShootingTypes.PISTOL
        return Shoot.ShootingTypes.AK47
    
    def startRowAction(self):
        print("taking first photo")
        print("عندما تكون جاهز اشتبككككك")
        #if(sr==None):
        #    sr = Shoot.ShootingResults("./results/"+sharedGUI.TEAM_NAME+"/", sharedGUI.NO_OF_TARGETS, shooting_type=encode_shooting_type(sharedGUI.SHOOTING_TYPE))
        self.sr.begin_shooting()

    
    def endRowAction(self):
        res= self.sr.end_shooting()
        print(res)
        return res
        '''
        print("taking second photo")
        print("returning results ....")
        return  (
                    [
                        2,
                        5,
                        0,
                        1,
                        3,
                        4,
                        4,
                        4,
                        4,
                        4,
                        4,
                        4,
                        4,
                        4,
                    ],
                    [
                        'images\\main.gif',
                        'images\\main1.gif',
                        'images\\main2.gif',
                        'images\\main3.gif',
                        'images\\main4.gif',
                        'images\\main1.gif',
                        'images\\main1.gif',
                        'images\\main1.gif',
                        'images\\main1.gif',
                        'images\\main1.gif',
                        'images\\main1.gif',
                        'images\\main1.gif',
                        'images\\main1.gif',
                        'images\\main1.gif',
                    ]
                )
            '''
    