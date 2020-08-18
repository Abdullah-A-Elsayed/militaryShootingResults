import os
import sys
import re
import cv2
import imutils
import datetime
import pytesseract
import numpy as np
from PIL import Image
from pylab import rcParams
from autocorrect import spell
from matplotlib import pyplot as plt
from skimage.filters import threshold_local
from imutils.object_detection import non_max_suppression
import configparser
def rel_to_abs(rel_path, current_file):
    dirname = os.path.dirname(current_file)
    return os.path.join(dirname, rel_path)
def imwrite_unicode(dirname, file_name, frame):
    #dirname = 'C:\\Users\\Abdallah Reda\\Downloads\\CVC-19-Documnet-Wallet-\\BackEnd\\visionapp\\Natinal_ID\\src\\project\\results\\السرية الثامنة\\'
    script_path = os.getcwd()
    os.chdir(dirname)
    #frame = cv2.imread('C:\\Users\\Abdallah Reda\\Downloads\\CVC-19-Documnet-Wallet-\\BackEnd\\visionapp\\Natinal_ID\\50.jpg')
    cv2.imwrite(file_name, frame)
    os.chdir(script_path)

config = configparser.ConfigParser()
config.read('conf.ini')
def get_configuration(section, key):
    return config[section][key]







