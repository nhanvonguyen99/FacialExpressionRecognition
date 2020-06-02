import time
import cv2
import numpy as np
import face_helper
import facs_helper
from imutils.video import VideoStream
import joblib


def main():
    # Label Facial Action Units (AUs) and Basic Emotions.
    global neutralFeatures

    dict_emotion = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    # Font size for text on video
    font_size = 0.6
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Initialize Dlib
    face_op = face_helper.faceUtil()

    # Facial landmarks
    vec = np.empty([68, 2], dtype=int)

    tol = 5 # Tolerance for setting neutral expression profile. Verifies eye and ear separation

    # Reduces images size for image processing.
    scaleFactor = 0.4

    # Scale up at the end for viewing.
    scaleUp = 3 / 4

    # Position of text on video when face is detected.
    pos_emotion = (np.arange(25, 225, 25) * scaleUp).astype(int)

    # SVM model path.
    load_file = 'model/decision_tree_model.sav'

    # SVM model
    model = joblib.load(load_file)

    # initialize the video stream and allow the camera sensor to warn up
    print("[INFO] camera sensor warming up...")
    vs = VideoStream(usePiCamera=False).start()
    time.sleep(5.0)
    neutralBool = False

    emotion = 6
    # loop over the frames from the video stream
    while True:
        frame = vs.read()
        small_frame = cv2.resize(frame, (256, 256))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        # Get facial landmarks and position of face on image.
        vec, point, face_bool = face_op.get_vec(gray)
        if face_bool:
            # Get facial features.
            feat = facs_helper.facialActions(vec, gray)
            newFeatures = feat.detectFeatures()
            if not neutralBool:
                neutralBool, neutralFeatures \
                    = face_op.set_neutral(feat, newFeatures, neutralBool, tol)
            else:
                # Just reshape variables.
                facialMotion = np.asarray(feat.FaceFeatures(neutralFeatures, newFeatures)).tolist()
                emotion = model.predict([facialMotion])[0]

        # Increase size of frame for viewing.
        big_frame = cv2.resize(small_frame, (0, 0), fx=scaleUp * 1 / scaleFactor, fy=scaleUp * 1 / scaleFactor)

        for idxJ, dd in enumerate(dict_emotion):
            cv2.putText(big_frame, dd, (380, pos_emotion[idxJ]), font, font_size, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(big_frame, dict_emotion[emotion], (380, pos_emotion[emotion]), font, font_size, (255, 0, 0), 2,
                    cv2.LINE_AA)

        cv2.imshow("Frame", big_frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
