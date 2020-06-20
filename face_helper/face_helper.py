"""Module for detecting faces, calculating facial landmarks, and setting the 
neutral expression."""

import cv2
import dlib
import numpy as np


class faceUtil:
    """Utility for facial analysis.
    
    Attributes:
        predictor_path (str): Path for dlib facial landmark predictor.
        detector: Dlib facial detector. Returns location of face.
        predictor: Gets 68 landmarks of detected face.
        vec (int): Holds landmarks of detected face. 
        neutralFeatures (float): Neutral facial features of face.
    """

    def __init__(self, predictor_path="./face_helper/shape_predictor_68_face_landmarks.dat"):
        self.predictor_path = predictor_path
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)
        self.vec = np.empty([68, 2], dtype=int)
        self.neutralFeatures = []

    def get_vec(self, image):
        """Get facial landmarks of face.
        
        Returns:
            vec (int): 68 landmarks of face.
            center (int): Coordinates of center of face.
            face_bool (bool): True if face detected. Otherwise, false."""

        detector = self.detector(image, 1)  # detector includes rectangle coordinates of face
        center = []  # Center coordinates of face.
        if np.any(detector):
            face_bool = True
            for k, d in enumerate(detector):
                shape = self.predictor(image, d)
                # Populate facial landmarks.
                for i in range(shape.num_parts):
                    self.vec[i][0] = shape.part(i).x
                    self.vec[i][1] = shape.part(i).y
                    # Identify landmarks with filled-in green circles on image.
                    cv2.circle(image, (self.vec[i][0], self.vec[i][1]), 1, (0, 255, 0))

                x = d.left()  # Column coordinate - Top left corner of detected face.
                y = d.top()  # Row coordinate - Top left corner of detected face.
                w = d.right() - x  # Width.
                h = d.bottom() - y  # Height.

                center = np.array((x + int(w / 2), y + int(h / 2)))

        else:
            face_bool = False
        return self.vec, center, face_bool

    def set_neutral(self, feat, newFeatures, neutralBool, tol):
        """Set neutral expression of detected face.
        
        In this script, facial emotion is detected based on displacement from 
        a neutral facial position to an emotional position. The subject must 
        initialize the robot with their neutral or blank facial expression
        for the facial actions to be detected properly.
        
        Args:
            neutralBool: Set neutral face or not 
            feat: Class for analyzing facial features. Used for checking face looks at camera.
            newFeatures: Facial features, candidates neutral expression.
            tol (int): Tolerance for how much head may be turned from straight-on portrait.
        
        Returns:
            neutralBool: True if face is looking directly at the camera. False, otherwise.
            neutralFeaturesUpper (float): Neutral facial features of upper face.
            neutralFeaturesLower (float): Neutral facial features of lower face.
        """
        if not neutralBool:
            jawBool, eyeBool = feat.checkProfile(tol)  # Check if the face is looking directly at the camera.
            if jawBool and eyeBool:
                self.neutralFeatures = newFeatures
                neutralBool = True
        return neutralBool, self.neutralFeatures

    def face_detect(self, image, face_bool):
        """Check if face is detected."""
        detector = self.detector(image, 1)  # detector includes rectangle coordinates of face
        if np.any(detector):
            face_bool = True
        return face_bool
