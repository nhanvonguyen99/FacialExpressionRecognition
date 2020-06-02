from sklearn.svm import SVC
import numpy as np
import glob
import os
import cv2
import face_helper
import facs_helper
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import argparse


def get_emotion(filename):
    emotion = filename[3: 5]
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


class FacePrepare:
    def __init__(self, faceFolder="dataset"):
        self.faceFolder = faceFolder
        self.faceUtil = face_helper.faceUtil()
        self.images = []
        self.labels = []

    def process(self):
        personName = "None"
        neutralFeatures = [], []
        for f in glob.glob(os.path.join(self.faceFolder, "*.tiff")):
            faceUtil = face_helper.faceUtil()
            print("Processing file: {}".format(f))
            currentPersonName = f.replace("dataset/", "")[:2]
            emotion = get_emotion(f.replace("dataset/", ""))
            if personName == "None" or personName != currentPersonName:
                personName = currentPersonName
                neutralImagePath = f[:11] + "NE" + ".tiff"
                image = cv2.imread(neutralImagePath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                vec, center, face_bool = faceUtil.get_vec(image)
                if face_bool:
                    feat = facs_helper.facialActions(vec, image)
                    neutralFeatures = feat.detectFeatures()

            image = cv2.imread(f)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            vec, center, face_bool = faceUtil.get_vec(image)
            if face_bool:
                feat = facs_helper.facialActions(vec, image)
                newFeatures = feat.detectFeatures()
                facialMotion = np.asarray(feat.FaceFeatures(neutralFeatures, newFeatures)).tolist()
                self.images.append(facialMotion)
                self.labels.append(emotion)

        return self.images, self.labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--load-data", type=bool, default=False,
                    help="True if load data from file, False if training data from dataset")
    ap.add_argument("-m", "--model-name", type=int, default=0,
                    help="Model name:\n\t0: SVM\n\t1: Gaussian naive bayes\n\tOther: Decision tree\n")
    ap.add_argument("-s", "--save-dataset", type=bool, default=True,
                    help="True if you want to save data to file")

    args = vars(ap.parse_args())

    modelName = args["model_name"]
    loadData = args["load_data"]
    if loadData:
        images = joblib.load("data_save/images.sav")
        labels = joblib.load("data_save/labels.sav")
    else:
        faceFolder = FacePrepare()
        images, labels = faceFolder.process()
        saveData = args["save_dataset"]
        if saveData:
            joblib.dump(images, "data_save/images.sav")
            joblib.dump(labels, "data_save/labels.sav")

    if modelName == 0:
        model = SVC(kernel="linear")
        path = "model/svm_linear_model.sav"
    elif modelName == 1:
        model = GaussianNB()
        path = "model/gaussian_naive_bayes_model.sav"
    else:
        model = DecisionTreeClassifier()
        path = "model/decision_tree_model.sav"

    model.fit(images, labels)
    joblib.dump(model, path)
    print("Model is saved at ", path)


if __name__ == "__main__":
    main()
