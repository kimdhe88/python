import MakeImageCV
import cv2
import numpy as np

import os
targetDir = "/home/hun/lab/python3.6.5/data/org"
resultDir = "/home/hun/lab/python3.6.5/data/result"
backgroundImagePath = "/home/hun/lab/python3.6.5/data/background/background_003.jpg"
#bgimg = cv2.imread(backgroundImagePath)  # mandrill reference image from USC SIPI
bgimg = cv2.imread(backgroundImagePath, 0)

imc = MakeImageCV.ImageController()
targetPath=""

#contrastList = [-100.0, -50.0, 0.0, 1.0, 50.0, 100.0]
contrastList = [0.0, 1.0, 50.0, 100.0]
brightnessList = [-100.0, -50.0, 0.0, 50.0, 100.0]
#contrastList=[0.0]
#brightnessList=[0.0]

for targetName in os.listdir(targetDir):
    if targetName.endswith(".jpg"):
        targetName
        targetPath = targetDir + '/' + targetName
        #targetImage = cv2.imread(targetPath)  # mandrill reference image from USC SIPI
        targetImage = cv2.imread(targetPath, 0)  # mandrill reference image from USC SIPI
        #imc.MakeImage(targetImage = targetImage, resultDir = resultDir, resultName = targetName)
        #imc.MakeImage(targetImage = targetImage, backgroundImage = bgimg, resultDir = resultDir, resultName = targetName, alpha = 1, beta = 1)
        imc.MakeImage(targetImage = targetImage, backgroundImage = bgimg, resultDir = resultDir, resultName = targetName \
                    , contrastList = contrastList , brightnessList = brightnessList \
                    , widthShiftCount = 1, heightShiftCount = 1)
