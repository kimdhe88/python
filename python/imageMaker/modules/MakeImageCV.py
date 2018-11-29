import cv2
import sys, os

class ImageController():
    def __init__(self):

        self.widthShiftSize = 0
        self.HeightShiftSize = 0
        self.isGrayscale = False

    def MakeImage(self, resultDir = None, resultName = None, targetImage = None, backgroundImage = None , \
                    contrastList = [1], brightnessList = [1], widthShiftCount = 1, heightShiftCount = 1):

        if targetImage is None:
            print("[Err] The target image is NULL.")
            quit()
        else:
            if len(targetImage.shape) == 2:
                self.isGrayscale = True

        if type(contrastList) is not list:
            print("The alpha value must be entered as a list type.")
            quit()
        if type(brightnessList) is not list:
            print("The beta value must be entered as a list type.")
            quit()

        mergeImage = None
        outImage = None

        fileName = os.path.splitext(os.path.basename(resultName))[0]
        fileExtensions = os.path.splitext(os.path.basename(resultName))[1]

        if backgroundImage is None:
            widthShiftCount = 1
            heightShiftCount = 1
        else:
            if self.isGrayscale:
                if len(backgroundImage.shape) != 2:
                    print("[Err] The target image and the background image have different color types.")
                    quit()
            else:
                if len(backgroundImage.shape) != 3:
                    print("[Err] The target image and the background image have different color types.")
                    quit()

            if widthShiftCount != 1:
                if widthShiftCount < 1:
                    widthShiftCount = 1
                self.widthShiftSize = int((backgroundImage.shape[1] - targetImage.shape[1]) / (widthShiftCount - 1))
            if heightShiftCount != 1:
                if heightShiftCount < 1:
                    heightShiftCount = 1
                self.HeightShiftSize = int((backgroundImage.shape[0] - targetImage.shape[0]) / (heightShiftCount - 1))

        #merge image
        for width in range(widthShiftCount):
            for height in range(heightShiftCount):
                widthOffset = self.widthShiftSize * width
                heightOffset = self.HeightShiftSize * height

                if backgroundImage is not None:
                    mergeImage = backgroundImage.copy()
                    mergeImage[heightOffset:heightOffset + targetImage.shape[0], widthOffset:widthOffset + targetImage.shape[1]] = targetImage
                else:
                    mergeImage = targetImage.copy()

                for contrast in contrastList:
                    for brightness in brightnessList:
                        resultPath = resultDir + '/' + fileName + ("_x_%s" % width) + ("_y_%s" % height) + ("_c_%s" % contrast) + ("_b_%s" % brightness) + fileExtensions
                        print(resultPath)
                        #outImage = cv2.addWeighted( mergeImage, contrast, 0, 0, brightness)
                        outImage = self.ApplyContrastBrightness(mergeImage,contrast,brightness)
                        cv2.imwrite(resultPath, outImage)

    def ApplyContrastBrightness(self, image = None, contrast = 0, brightness = 0):
        if image is None:
            print("[Err] The image is NULL.")
            quit()

        print("contrast : %s" % contrast)
        print("brightness : %s" % brightness)

        if contrast < -100:
            contrast = -100
        elif contrast >= 100:
            contrast = 99

        if brightness < 0:
            brightness = 0

        if contrast < 0:
            alpha = (100 + contrast) / 100
        elif contrast >= 0:
            alpha = 100 / (100 - contrast)


        alpha = round(alpha,2)


        #print(image.dtype)
        #print(image)
        image = image.astype('int32')
        image = 128 + alpha * ( image - 128) + brightness
        image[image < 0] = 0
        image[image > 255] = 255
        image = image.astype('uint8')
        #print(image)
        return image
