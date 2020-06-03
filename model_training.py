from sklearn.svm import SVC
import numpy as np
import glob
import os
import cv2
import face_helper
import facs_helper
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

facsTop = np.array([1., 2., 4., 5., 6., 7.])
facsBtm = np.array([9., 10., 12., 15., 16., 20., 23., 24., 25., 26., 27.])


def get_upper_aus(filename):
    emotion = filename[3: 5]
    if emotion == "HA":
        return [4]
    elif emotion == "SA":
        return [0, 2]
    elif emotion == "SU":
        return [0, 1, 3]
    elif emotion == "FE":
        return [0, 1, 2, 3, 5]
    elif emotion == "AN":
        return [2, 3, 5]
    return [-1]


def get_lower_aus(filename):
    emotion = filename[3: 5]
    if emotion == "HA":
        return [2]
    elif emotion == "SA":
        return [3]
    elif emotion == "SU":
        return [4, 9]
    elif emotion == "FE":
        return [3]
    elif emotion == "AN":
        return [6]
    elif emotion == "DI":
        return [3, 4]
    return [-1]


class FacePrepare:
    def __init__(self, faceFolder="dataset"):
        self.faceFolder = faceFolder
        self.faceUtil = face_helper.faceUtil()
        self.upper_face_features = []
        self.lower_face_features = []
        self.upper_face_labels = []
        self.lower_face_labels = []

    def process(self):
        personName = "None"
        neutralUpperFeatures, neutralLowerFeature = [], []
        for f in glob.glob(os.path.join(self.faceFolder, "*.tiff")):
            faceUtil = face_helper.faceUtil()
            print("Processing file: {}".format(f))
            currentPersonName = f.replace("dataset/", "")[:2]
            upperAu = get_upper_aus(f.replace("dataset/", ""))
            lowerAu = get_lower_aus(f.replace("dataset/", ""))
            if personName == "None" or personName != currentPersonName:
                personName = currentPersonName
                neutralImagePath = f[:11] + "NE" + ".tiff"
                image = cv2.imread(neutralImagePath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                vec, center, face_bool = faceUtil.get_vec(image)
                if face_bool:
                    feat = facs_helper.facialActions(vec, image)
                    neutralUpperFeatures = feat.detectUpperFeatures()
                    neutralLowerFeature = feat.detectLowerFeatures()

            image = cv2.imread(f)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            vec, center, face_bool = faceUtil.get_vec(image)
            if face_bool:
                feat = facs_helper.facialActions(vec, image)
                newUpperFeatures = feat.detectUpperFeatures()
                newLowerFeatures = feat.detectLowerFeatures()
                upperFacialMotion = np.asarray(feat.UpperFaceFeatures(neutralUpperFeatures, newUpperFeatures)).tolist()
                lowerFacialMotion = np.asarray(feat.LowerFaceFeatures(neutralLowerFeature, newLowerFeatures)).tolist()
                if -1 not in upperAu:
                    self.upper_face_features.append(upperFacialMotion)
                    self.upper_face_labels.append(upperAu)
                if -1 not in lowerAu:
                    self.lower_face_features.append(lowerFacialMotion)
                    self.lower_face_labels.append(lowerAu)
        return self.upper_face_features, self.upper_face_labels, self.lower_face_features, self.lower_face_labels


def main():
    upper_face_features = joblib.load("dataset/upper_face_features.sav")
    upper_face_labels = joblib.load("dataset/upper_face_labels.sav")
    upper_face_features = np.asarray(upper_face_features)
    upper_face_labels = np.asarray(upper_face_labels)

    lower_face_features = joblib.load("dataset/lower_face_features.sav")
    lower_face_labels = joblib.load("dataset/lower_face_labels.sav")
    lower_face_features = np.asarray(lower_face_features)
    lower_face_labels = np.asarray(lower_face_labels)

    mlb = MultiLabelBinarizer()
    upper_face_labels = mlb.fit_transform(upper_face_labels)
    upper_face_classes = mlb.classes_
    mlb1 = MultiLabelBinarizer()
    lower_face_labels = mlb1.fit_transform(lower_face_labels)
    lower_face_classes = mlb1.classes_

    joblib.dump(upper_face_classes, "data_save/upper_face_classes.sav")
    joblib.dump(lower_face_classes, "data_save/lower_face_classes.sav")

    subClf = SVC(kernel="linear")
    clf = OneVsRestClassifier(estimator=subClf)
    clf.fit(lower_face_features, lower_face_labels)

    subClf1 = SVC(kernel="linear")
    clf1 = OneVsRestClassifier(estimator=subClf1)
    clf1.fit(upper_face_features, upper_face_labels)

    joblib.dump(clf, "model/lower_svm_linear_model.sav")
    joblib.dump(clf1, "model/upper_svm_linear_model.sav")


if __name__ == "__main__":
    main()
