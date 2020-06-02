import time
import imutils
import cv2
import numpy as np
import face_helper
import facs_helper
import tf_helper
from imutils.video import VideoStream
from sklearn.svm import SVC
import joblib


def main():
    # Label Facial Action Units (AUs) and Basic Emotions.
    dict_upper = ['AU1: Inner Brow Raiser', 'AU2: Outer Brow Raiser', 'AU4: Brow Lowerer', 'AU5: Upper Lid Raiser',
                  'AU6: Cheek Raiser', 'AU7: Lid Tightener']
    dict_lower = ['AU9: Nose Wrinkler', 'AU10: Upper Lip Raiser', 'AU12: Lip Corner Puller',
                  'AU15: Lip Corner Depressor',
                  'AU17: Chin Raiser', 'AU20: Lip Stretcher', 'AU23: Lip Tightener', 'AU24: Lip Pressor',
                  'AU25: Lips Part',
                  'AU26: Jaw Drop', 'AU27: Mouth Stretch']
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

    # Get center of image.
    width = 640
    height = 480
    centerFixed = np.array((int(width * scaleFactor / 2), int(height * scaleFactor / 2)))

    # Position of text on video when face is detected.
    pos_lower = (np.arange(175, 450, 25) * scaleUp).astype(int)
    pos_upper = (np.arange(25, 175, 25) * scaleUp).astype(int)
    pos_emotion = (np.arange(25, 225, 25) * scaleUp).astype(int)

    # Tensorflow model path.
    load_file = 'model/svm_model.sav'

    model = joblib.load(load_file)

    # initialize the video stream and allow the camera sensor to warn up
    print("[INFO] camera sensor warming up...")
    vs = VideoStream(usePiCamera=False).start()
    time.sleep(5.0)
    neutralBool = False

    idxFacsLow, idxFacsUp = [], []
    emotion = 6
    # loop over the frames from the video stream
    while True:
        face_bool = True
        frame = vs.read()

        small_frame = cv2.resize(frame, (0, 0), fx=scaleFactor, fy=scaleFactor)
        # Get facial landmarks and position of face on image.
        vec, point, face_bool = face_op.get_vec(small_frame, centerFixed, face_bool)
        if face_bool:
            # Get facial features.
            feat = facs_helper.facialActions(vec, small_frame)
            newFeaturesUpper = feat.detectFeatures()
            newFeaturesLower = feat.detectLowerFeatures()
            if not neutralBool:
                neutralBool, neutralFeaturesUpper, neutralFeaturesLower \
                    = face_op.set_neutral(feat, newFeaturesUpper, newFeaturesLower, neutralBool, tol)
            else:
                # Just reshape variables.
                facialMotionUp = np.asarray(feat.UpperFaceFeatures(neutralFeaturesUpper, newFeaturesUpper))
                facialMotionLow = np.asarray(feat.LowerFaceFeatures(neutralFeaturesLower, newFeaturesLower))
                facialMotion = np.concatenate((facialMotionUp, facialMotionLow)).tolist()
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
