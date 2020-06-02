from sklearn.svm import SVC
import numpy as np
import glob
import os
import cv2
import face_helper
import facs_helper
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


def get_emotion(filename):
    emotion = filename[3: 5]
    if emotion == "AN":
        return 0
    elif emotion == "DI":
        return 1
    elif emotion == "FE":
        return 2
    elif emotion == "HA":
        return 3
    elif emotion == "SA":
        return 4
    elif emotion == "SU":
        return 5
    return 6


class FacePrepare:
    def __init__(self, faceFolder="dataset"):
        self.faceFolder = faceFolder
        self.faceUtil = face_helper.faceUtil()
        self.images = []
        self.labels = []

    def process(self):
        personName = "None"
        neutralFeaturesUpper, neutralFeaturesLower = [], []
        for f in glob.glob(os.path.join(self.faceFolder, "*.tiff")):
            faceUtil = face_helper.faceUtil()
            print("Processing file: {}".format(f))
            currentPersonName = f.replace("dataset/", "")[:2]
            emotion = get_emotion(f.replace("dataset/", ""))
            if personName == "None" or personName != currentPersonName:
                personName = currentPersonName
                neutralImagePath = f[:11] + "NE" + ".tiff"
                image = cv2.imread(neutralImagePath)
                vec, center, face_bool = faceUtil.get_vec(image)
                if face_bool:
                    feat = facs_helper.facialActions(vec, image)
                    neutralFeaturesUpper = feat.detectFeatures()
                    neutralFeaturesLower = feat.detectLowerFeatures()

            image = cv2.imread(f)
            vec, center, face_bool = faceUtil.get_vec(image)
            if face_bool:
                feat = facs_helper.facialActions(vec, image)
                newFeaturesUpper = feat.detectFeatures()
                newFeaturesLower = feat.detectLowerFeatures()
                facialMotionUp = np.asarray(feat.UpperFaceFeatures(neutralFeaturesUpper, newFeaturesUpper))
                facialMotionLow = np.asarray(feat.LowerFaceFeatures(neutralFeaturesLower, newFeaturesLower))
                facialMotion = np.concatenate((facialMotionUp, facialMotionLow)).tolist()
                self.images.append(facialMotion)
                self.labels.append(emotion)

        return self.images, self.labels


def main():
    faceFolder = FacePrepare()
    images, labels = faceFolder.process()

    model = SVC(kernel="poly")
    model.fit(images, labels)
    joblib.dump(model, "model/svm_poly_model.sav")
    pred = model.predict(images)
    print(accuracy_score(labels, pred))


if __name__ == "__main__":
    main()
