import glob
import os
import numpy as np
import cv2
from face_helper import faceUtil
from facs_helper import facialActions


class FacePrepare:
    def __init__(self, faceFolder="dataset"):
        self.__faceFolder = faceFolder
        self.__faceUtil = faceUtil()
        self.__neutralFeatures = {}
        self.images_ = []
        self.labels_ = []

        self.__get_neural_feature()

    def __get_neural_feature(self):
        for f in glob.glob(os.path.join(self.__faceFolder, "*.tiff")):
            personName = f[8:10]
            if personName not in self.__neutralFeatures.keys():
                neuralImagePath = glob.glob(os.path.join(self.__faceFolder, "{0}.NE*.tiff".format(personName)))[0]
                image = cv2.imread(neuralImagePath)
                vec, center, face_bool = self.__faceUtil.get_vec(image)
                if face_bool:
                    feat = facialActions(vec, image)
                    neutralFeatures = feat.detectFeatures()
                    self.__neutralFeatures[personName] = neutralFeatures

    @staticmethod
    def filename2emotion(filename):
        emotion = filename[11:13]
        if emotion == "AN":
            return 1
        elif emotion == "DI":
            return 2
        elif emotion == "FE":
            return 3
        elif emotion == "HA":
            return 4
        elif emotion == "SA":
            return 5
        elif emotion == "SU":
            return 6
        return 0

    def process(self):
        for f in glob.glob(os.path.join(self.__faceFolder, "*.tiff")):
            print("Processing file: {0}".format(f))
            personName = f[8:10]
            emotion = self.filename2emotion(f)
            neutralFeatures = self.__neutralFeatures[personName]
            image = cv2.imread(f)
            vec, center, faceBool = self.__faceUtil.get_vec(image)
            if faceBool:
                feat = facialActions(vec, image)
                newFeatures = feat.detectFeatures()
                facialMotion = np.asarray(feat.FaceFeatures(neutralFeatures, newFeatures), dtype="float64").tolist()
                self.images_.append(facialMotion)
                self.labels_.append(emotion)

        return self.images_, self.labels_
