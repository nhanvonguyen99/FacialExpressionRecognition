from os import path

import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from face_prepare import FacePrepare


def main():
    if path.exists("data_save/images.sav") and path.exists("data_save/labels.sav"):
        images = joblib.load("data_save/images.sav")
        labels = joblib.load("data_save/labels.sav")
    else:
        facePrepare = FacePrepare()
        images, labels = facePrepare.process()
        joblib.dump(images, "data_save/images.sav")
        joblib.dump(labels, "data_save/labels.sav")
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=2)
    model = SVC(kernel="rbf")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("SVM:", accuracy_score(y_test, pred))

    model_save = SVC(kernel="rbf")
    model_save.fit(images, labels)
    joblib.dump(model_save, "model/svm_rbf_model.sav")

    # facePrepare.process_features_selection("SVC")
    # facePrepare.process_features_selection("GaussianNB")
    # facePrepare.process_features_selection("DecisionTreeClassifier")


if __name__ == "__main__":
    main()
