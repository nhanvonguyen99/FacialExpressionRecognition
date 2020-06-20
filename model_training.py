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
    # Cs = []
    # accuracies = []
    #
    # accuracies1 = []
    #
    # accuracies2 = []
    #
    # accuracies3 = []
    #
    # i = 0.1
    # while i < 3.1:
    #     model = SVC(kernel="rbf", C=i)
    #     model.fit(X_train, y_train)
    #     pred = model.predict(X_test)
    #     accuracies.append(accuracy_score(y_test, pred))
    #
    #     model1 = SVC(kernel="linear", C=i)
    #     model1.fit(X_train, y_train)
    #     pred = model1.predict(X_test)
    #     accuracies1.append(accuracy_score(y_test, pred))
    #
    #     model2 = SVC(kernel="poly", C=i)
    #     model2.fit(X_train, y_train)
    #     pred = model2.predict(X_test)
    #     accuracies2.append(accuracy_score(y_test, pred))
    #
    #     model3 = SVC(kernel="sigmoid", C=i)
    #     model3.fit(X_train, y_train)
    #     pred = model3.predict(X_test)
    #     accuracies3.append(accuracy_score(y_test, pred))
    #
    #     Cs.append(i)
    #     i += 0.1
    #
    # import matplotlib.pyplot as plt
    # plt.plot(Cs, accuracies, "r", label="rbf")
    # plt.plot(Cs, accuracies1, "b", label="linear")
    # plt.plot(Cs, accuracies2, "g", label="polynomial")
    # plt.plot(Cs, accuracies3, "y", label="sigmoid")
    #
    # plt.xlabel("Regularization parameter (C)")
    # plt.ylabel("Accuracy")
    # plt.title("Regularization parameter regarding to accuracy")
    #
    # plt.legend()
    #
    # plt.show()

    model = SVC(kernel="rbf")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("SVM:", accuracy_score(y_test, pred))

    from sklearn.metrics import plot_confusion_matrix
    import matplotlib.pyplot as plt

    # Plot non-normalized confusion matrix

    class_names = ["Neutral", "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]

    dist = plot_confusion_matrix(model, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 display_labels=class_names, normalize="true")
    dist.ax_.set_title("Expression Classes - Recognition Rate")

    plt.show()
    model_save = SVC(kernel="rbf", C=2.3)
    model_save.fit(images, labels)
    joblib.dump(model_save, "model/svm_rbf_model.sav")

    # facePrepare.process_features_selection("SVC")
    # facePrepare.process_features_selection("GaussianNB")
    # facePrepare.process_features_selection("DecisionTreeClassifier")


if __name__ == "__main__":
    main()
