import joblib
import argparse
from sklearn.model_selection import train_test_split
from face_prepare import FacePrepare
import tensorflow as tf
from tensorflow import keras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--load-data", type=bool, default=False,
                    help="True if load data from file, False if training data from dataset")
    ap.add_argument("-s", "--save-dataset", type=bool, default=True,
                    help="True if you want to save data to file")

    args = vars(ap.parse_args())

    loadData = args["load_data"]
    if loadData:
        images = joblib.load("data_save/images_temp.sav")
        labels = joblib.load("data_save/labels_temp.sav")
    else:
        faceFolder = FacePrepare()
        images, labels = faceFolder.process()
        saveData = args["save_dataset"]
        if saveData:
            joblib.dump(images, "data_save/images_temp.sav")
            joblib.dump(labels, "data_save/labels_temp.sav")

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=2)
    #  print(len(X_train[0]))
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(len(images[0]),)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(7)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=1000)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

    print('\nTest accuracy:', test_acc)

    model.fit(images, labels, epochs=1000)
    model.save("model/dnn")


if __name__ == "__main__":
    main()
