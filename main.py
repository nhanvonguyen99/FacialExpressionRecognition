import time
import cv2
import numpy as np
import face_helper
import facs_helper
from imutils.video import VideoStream
import model_helper
import facs_to_emotion


def main():

    # Label Facial Action Units (AUs) and Basic Emotions.
    global neutralFeatures, idxFacsLow, idxFacsUp, neutralFeaturesUpper, neutralFeaturesLower

    # Label Facial Action Units (AUs) and Basic Emotions.
    dict_upper = ['AU1: Inner Brow Raiser', 'AU2: Outer Brow Raiser', 'AU4: Brow Lowerer', 'AU5: Upper Lid Raiser',
                  'AU6: Cheek Raiser', 'AU7: Lid Tightener']
    dict_lower = ['AU9: Nose Wrinkler', 'AU10: Upper Lip Raiser', 'AU12: Lip Corner Puller',
                  'AU15: Lip Corner Depressor', 'AU17: Chin Raiser', 'AU20: Lip Stretcher', 'AU23: Lip Tightener',
                  'AU24: Lip Pressor', 'AU25: Lips Part', 'AU26: Jaw Drop', 'AU27: Mouth Stretch']
    dict_emotion = ['Thinking...', 'Happiness', 'Sad', 'Surprise', 'Fear', 'Anger', 'Disgust']

    # Font size for text on video
    font_size = 0.6
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Initialize Dlib
    face_op = face_helper.faceUtil()

    tol = 5 # Tolerance for setting neutral expression profile. Verifies eye and ear separation

    # Reduces images size for image processing.
    scaleFactor = 0.4

    # Scale up at the end for viewing.
    scaleUp = 3 / 4

    width = 640
    height = 480
    centerFixed = np.array((int(width * scaleFactor / 2), int(height * scaleFactor / 2)))

    # Position of text on video when face is detected.
    pos_lower = (np.arange(175, 450, 25) * scaleUp).astype(int)
    pos_upper = (np.arange(25, 175, 25) * scaleUp).astype(int)
    pos_emotion = (np.arange(25, 225, 25) * scaleUp).astype(int)
    # Counts iterations when face is not found.
    iter_count = 0
    # Stores facial features of upper face.
    groundFeatUpper = []
    # Stores facial features of lower face.
    groundFeatLower = []
    # Stores change in facial features of upper face.
    facialMotionUpper = []
    # Stores change in facial features of lower face.
    facialMotionLower = []
    # Boolean flag for when neutral expresion is set.
    # model
    # model path.
    load_file_low = 'model/lower_svm_linear_model.sav'
    load_file_up = 'model/upper_svm_linear_model.sav'
    low_n_classes_path = "data_save/lower_face_classes.sav"
    up_n_classes_path = "data_save/upper_face_classes.sav"

    # Tensorflow model for lower and upper face.
    modelLow = model_helper.modelUtil(load_file_low, low_n_classes_path)
    modelUp = model_helper.modelUtil(load_file_up, up_n_classes_path)

    # initialize the video stream and allow the camera sensor to warn up
    print("[INFO] camera sensor warming up...")
    vs = VideoStream(usePiCamera=False).start()
    time.sleep(5.0)
    neutralBool = False

    emotion = 6
    idxFacsLow = []
    idxFacsUp = []
    # loop over the frames from the video stream
    while True:
        frame = vs.read()
        # Get facial landmarks and position of face on image.
        face_bool = False
        small_frame = cv2.resize(frame, (0, 0), fx=scaleFactor, fy=scaleFactor)
        vec, point, face_bool = face_op.get_vec(small_frame)
        if face_bool:
            # Get facial features.
            feat = facs_helper.facialActions(vec, small_frame)
            newFeaturesUpper = feat.detectUpperFeatures()
            newFeaturesLower = feat.detectLowerFeatures()
            if not neutralBool:
                neutralBool, neutralFeaturesUpper, neutralFeaturesLower \
                    = face_op.set_neutral(feat, newFeaturesUpper, newFeaturesLower, neutralBool, tol)
            else:
                # Just reshape variables.
                facialMotionUp = np.asarray(feat.UpperFaceFeatures(neutralFeaturesUpper, newFeaturesUpper))
                facialMotionLow = np.asarray(feat.LowerFaceFeatures(neutralFeaturesLower, newFeaturesLower))
                # Predict AUs with TF model.
                facsLow = modelLow.run(facialMotionLow)
                facsUp = modelUp.run(facialMotionUp)
                # Predict emotion based on AUs.
                feel = facs_to_emotion.facs2emotion(facsUp, facsLow)
                emotion = feel.declare()
                # Get index of AUs.
                idxFacsLow = facsLow
                idxFacsUp = facsUp
        # Increase size of frame for viewing.
        big_frame = cv2.resize(small_frame, (0, 0), fx=scaleUp * 1 / scaleFactor, fy=scaleUp * 1 / scaleFactor)
        # Show text on video.
        for idxJ, dd in enumerate(dict_upper):
            cv2.putText(big_frame, dd, (10, pos_upper[idxJ]), font, font_size, (255, 255, 255), 2, cv2.LINE_AA)
        for idxJ, dd in enumerate(dict_lower):
            cv2.putText(big_frame, dd, (10, pos_lower[idxJ]), font, font_size, (255, 255, 255), 2, cv2.LINE_AA)
        for idxJ, dd in enumerate(dict_emotion):
            cv2.putText(big_frame, dd, (380, pos_emotion[idxJ]), font, font_size, (255, 255, 255), 2, cv2.LINE_AA)

        # Write text on frame.
        if len(idxFacsLow) > 0:
            for ii in idxFacsLow:
                cv2.putText(big_frame, dict_lower[ii], (10, pos_lower[ii]), font, font_size, (255, 0, 0), 2,
                            cv2.LINE_AA)
        if len(idxFacsUp) > 0:
            for jj in idxFacsUp:
                cv2.putText(big_frame, dict_upper[jj], (10, pos_upper[jj]), font, font_size, (255, 0, 0), 2,
                            cv2.LINE_AA)
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
