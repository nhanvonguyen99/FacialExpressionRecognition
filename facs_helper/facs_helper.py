# -*- coding: utf-8 -*-
"""This module processes facial landmarks and converts facial landmarks to facial features."""

__name__ = 'facs_helper'

import cv2
import numpy as np


class facialActions:
    """Get facial landmarks and find displacements of facial landmarks over time to determine
    facial features.
    
    Args:
        vec (int): 68 facial landmarks.
        img: Image.
        
    Attributes:
        newFeatures (float): Facial displacements of key landmarks. 
        
        (The following are *coordinates* from slices of the array 'vec'.
        All are type 'int'.)
        brow_ell: Left eyebrow.
        brow_r: Right eyebrow.
        eye_ell: Left eye.
        eye_r: Right eye.
        nose_line: Coordinates from tip of nose to brow. 
        nose_arc: Coordinates outlining arc from one nostril to the other. 
        lip_tu: Top upper lip. 
        lip_bl: Bottom lower lip.
        lip_tl: Top lower lip.
        lip_bu: Bottom upper lip. 
        jaw: Outline of jaw from ear to ear. 
        
        (The following are not used in the robot, but may be used to process CK+ database.
        These variables are density of lines calculated from Canny edges divided by the 
        area in the window. All have type 'float.' )
        furrow: Density of lines from canny edges between brows. 
        wrinkle_ell: Density left of left eye. 
        wrinkle_r: Density right of right eye. 
        brow_ri: Density above right inner eyebrow.
        brow_li: Density above left inner eyebrow. 
        brow_ro: Density above right outer eyebrow.
        brow_lo: Density above left outer eyebrow. 
    """

    def __init__(self, vec, img):
        dist = 10
        dist_eye = 10
        dist_shift = 10
        dist_shift_brow = 10
        self.newFeatures = []

        # Declare key facial distances, ell means left, r is for right, u is for upper, b is for bottom
        # u is for upper, l is for lower, i is for inner, and o is for outer.
        self.brow_ell = vec[17:22, :]
        self.brow_r = vec[22:27, :]
        self.eye_ell = vec[36:42, :]
        self.eye_r = vec[42:48, :]
        self.nose_line = vec[27:31, :]
        self.nose_arc = vec[31:36, :]
        self.lip_tu = vec[48:54, :]
        self.lip_bl = vec[54:60, :]
        self.lip_tl = vec[60:64, :]
        self.lip_bu = vec[64:68, :]
        self.jaw = vec[0:17, :]

        # Regions of interest can detect wrinkles between the brow
        # and on the corner of the eye. These are transient 
        # features as young people do not have as many wrinkles as
        # older people. The Canny edge detector finds lines, and the
        # algorithm computes the density over the sample area. 
        roi = img[self.nose_line[0, 1] - dist:self.nose_line[0, 1] + dist,
              self.nose_line[0, 0] - dist: self.nose_line[0, 0] + dist]
        roi_ell = img[self.eye_ell[0, 1] - dist_eye:self.eye_ell[0, 1] + dist_eye,
                  self.eye_ell[0, 0] - dist_eye - dist_shift: self.eye_ell[0, 0] + dist_eye - dist_shift]
        roi_r = img[self.eye_r[3, 1] - dist_eye:self.eye_r[3, 1] + dist_eye,
                self.eye_r[3, 0] - dist_eye + dist_shift: self.eye_r[3, 0] + dist_eye + dist_shift]
        roi_brow_ri = img[self.brow_r[0, 1] - dist - dist_shift_brow:self.brow_r[0, 1] + dist - dist_shift_brow,
                      self.brow_r[0, 0] - dist: self.brow_r[0, 0] + dist]
        roi_brow_li = img[self.brow_ell[4, 1] - dist - dist_shift_brow:self.brow_ell[4, 1] + dist - dist_shift_brow,
                      self.brow_ell[4, 0] - dist: self.brow_ell[4, 0] + dist]
        roi_brow_ro = img[self.brow_r[4, 1] - dist - dist_shift_brow:self.brow_r[4, 1] + dist - dist_shift_brow,
                      self.brow_r[4, 0] - dist: self.brow_r[4, 0] + dist]
        roi_brow_lo = img[self.brow_ell[0, 1] - dist - dist_shift_brow:self.brow_ell[0, 1] + dist - dist_shift_brow,
                      self.brow_ell[0, 0] - dist: self.brow_ell[0, 0] + dist]
        canny = cv2.Canny(roi, 50, 200)
        canny_eye_r = cv2.Canny(roi_r, 50, 200)
        canny_eye_ell = cv2.Canny(roi_ell, 50, 200)
        canny_brow_ri = cv2.Canny(roi_brow_ri, 50, 200)
        canny_brow_li = cv2.Canny(roi_brow_li, 100, 200)
        canny_brow_ro = cv2.Canny(roi_brow_ro, 100, 200)
        canny_brow_lo = cv2.Canny(roi_brow_lo, 100, 200)
        self.furrow = np.sum((0 if canny is None else canny) / 255) / dist ** 2
        self.wrinkle_ell = np.sum((0 if canny_eye_ell is None else canny_eye_ell) / 255) / dist_eye ** 2
        self.wrinkle_r = np.sum((0 if canny_eye_r is None else canny_eye_r) / 255) / dist_eye ** 2
        self.brow_ri = np.sum((0 if canny_brow_ri is None else canny_brow_ri) / 255) / dist ** 2
        self.brow_li = np.sum((0 if canny_brow_li is None else canny_brow_li) / 255) / dist ** 2
        self.brow_ro = np.sum((0 if canny_brow_ro is None else canny_brow_ro) / 255) / dist ** 2
        self.brow_lo = np.sum((0 if canny_brow_lo is None else canny_brow_lo) / 255) / dist ** 2

    def detectFeatures(self):
        """Get upper facial features, which are displacements over time of facial landmarks.
        Refer to https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
        
        Returns: 
            D: Distance between eyebrows.
            blo: Height between outer corner of left eye and outer left eyebrow.
            bli: Height between inner corner of left eye and inner left eyebrow.
            bri: Height between outer corner of right eye and outer right eyebrow.
            bro: Height between inner corner of right eye and inner right eyebrow.
            hl1: Height of top left eyelid from pupil.
            hr1: Height of top right eyelid from pupil.
            hl2: Height of bottom left eyelid from pupil.
            hr2: Height of bottom right eyelid from pupil.
            hl3: Height of top left eyelid from pupil using alternate coordinate.
            hr3: Height of top right eyelid from pupil using alternate coordinate.
            bl: Distance from top left eyebrow to brim of nose.
            br: Distance from top right eyebrow to brim of nose.
            n_arc: Height from brim of nose to corner of nose.
            hl_0: Height of left eye from eyelid to eyelid.
            hr_0: Height of right eye from eyelid to eyelid.
            h1: Height of top lip from corner of mouth.
            h2: Height of bottom lip from corner of mouth.
            w: Width of mouth from corner to corner of lips.
            D_ell:  Height from left eye to left corner of mouth.
            D_r: Height from right eye to right corner of mouth.
            D_top: Height of top lip to bridge of nose.
            D_b: Height of bottom lip to bridge of nose.
        """
        D = abs(self.brow_r[0, 0] - self.brow_ell[4, 0])  # distance between eyebrows
        blo = abs((self.brow_ell[0, 1] + self.brow_ell[1, 1]) / 2 - self.eye_ell[
            0, 1])  # height between outer corner of left eye and outer left eyebrow
        bli = abs((self.brow_ell[4, 1] + self.brow_ell[3, 1]) / 2 - self.eye_ell[
            3, 1])  # height between inner corner of left eye and inner left eyebrow

        bri = abs((self.brow_r[0, 1] + self.brow_r[1, 1]) / 2 - self.eye_r[
            0, 1])  # height between outer corner of right eye and outer right eyebrow

        bro = abs((self.brow_r[4, 1] + self.brow_r[3, 1]) / 2 - self.eye_r[
            3, 1])  # height between inner corner of right eye and inner right eyebrow

        hl1 = (1 + abs(self.eye_ell[0, 1] - self.eye_ell[2, 1]))  # Height of top left eyelid from pupil
        hr1 = (1 + abs(self.eye_r[3, 1] - self.eye_r[1, 1]))  # Height of top right eyelid from pupil
        hl2 = (1 + abs(self.eye_ell[0, 1] - self.eye_ell[4, 1]))  # Height of bottom left eyelid from pupil
        hr2 = (1 + abs(self.eye_r[3, 1] - self.eye_r[5, 1]))  # Height of bottom right eyelid from pupil
        hl3 = (1 + abs(self.eye_ell[2, 1] - self.nose_line[0, 1]))
        hr3 = (1 + abs(self.eye_r[1, 1] - self.nose_line[0, 1]))

        bl = abs(self.brow_ell[2, 1] - self.nose_line[0, 1])  # distance from top left eyebrow to brim of nose
        br = abs(self.brow_r[2, 1] - self.nose_line[0, 1])  # distance from top right eyebrow to brim of nose
        n_arc = abs((self.nose_arc[0, 1] + self.nose_arc[4, 1]) / 2 - self.nose_line[
            0, 1])  # height from brim of nose to corner of nose
        hl_0 = abs((self.eye_ell[1, 1] + self.eye_ell[2, 1]) / 2 - (
                self.eye_ell[4, 1] + self.eye_ell[5, 1]) / 2)  # Height of left eye from eyelid to eyelid
        hr_0 = abs((self.eye_r[1, 1] + self.eye_r[2, 1]) / 2 - (
                self.eye_r[4, 1] + self.eye_r[5, 1]) / 2)  # Height of right eye from eyelid to eyelid

        h1 = abs(
            self.lip_tu[3, 1] - (self.lip_tu[0, 1] + self.lip_bl[0, 1]) / 2)  # Height of top lip from corner of mouth
        h2 = abs(self.lip_bl[3, 1] - (
                self.lip_tu[0, 1] + self.lip_bl[0, 1]) / 2)  # Height of bottom lip from corner of mouth
        w = abs(self.lip_tu[0, 0] - self.lip_bl[0, 0])  # Width of mouth from corner to corner of lips.
        D_ell = abs(self.lip_tu[0, 1] - self.nose_line[0, 1])  # Height from left eye to left corner of mouth.
        D_r = abs(self.lip_bl[0, 1] - self.nose_line[0, 1])  # Height from right eye to right corner of mouth.
        D_top = abs(self.lip_tu[3, 1] - self.nose_line[0, 1])  # Height of top lip to bridge of nose.
        D_b = abs(self.lip_bl[3, 1] - self.nose_line[0, 1])  # Height of bottom lip to bridge of nose.

        self.newFeatures = [D, blo, bli, bro, bri, hl1, hr1, hl2, hr2, self.furrow, self.wrinkle_ell, self.wrinkle_r,
                            bl, br, n_arc, hl_0, hr_0, self.brow_ri, self.brow_li, self.brow_ro, self.brow_lo, hl3, hr3,
                            h1, h2, w, D_ell, D_r, D_top, D_b]
        return self.newFeatures

    @staticmethod
    def FaceFeatures(old, new):
        """Motion of upper facial features comparing new frame to old frame.
        
        Not all values are returned for the robot. Canny edges, due to lighting, 
        were disrupting results. Performance on faces in the wild 
        improved with fewer arguments.
        
        Note that all displacements over time are scaled by the initial neutral position.
        This attempts to keep the analysis consistent for analyzing faces of different
        size and keeping the analysis scale invariant when the face is closer or farther away.
        It works okay, but the distance of the face does matter because the CK+ database
        provides faces all at the same distance from the camera. 
        
        Args:
            old: Upper static facial features from function detectFeatures.
            new: Upper static facial features from function detectFeatures.
        
        Returns:
            (all floats)
            r_D: Change in Distance between eyebrows.
            r_blo: Change in height between outer corner of left eye and outer left eyebrow.
            r_bli: Change in height between inner corner of left eye and inner left eyebrow.
            r_bri: Change in height between outer corner of right eye and outer right eyebrow.
            r_bro: Change in height between inner corner of right eye and inner right eyebrow.
            r_hl1: Change in height of top left eyelid from pupil.
            r_hr1: Change in height of top right eyelid from pupil.
            r_hl2: Change in height of bottom left eyelid from pupil.
            r_hr2: Change in height of bottom right eyelid from pupil.
            r_hl3: Change in height of bottom left eyelid from pupil.
            r_hr3: Change in height of bottom right eyelid from pupil.
            r_el: Change in left eye height. 
            r_er: Change in right eye height. 
            r_furrow: Change in density of lines from canny edges between brows. 
            r_wrinkle_ell: Change in density left of left eye. 
            r_wrinkle_r: Change in density right of right eye. 
            r_bl: Change in distance from top left eyebrow to brim of nose.
            r_br: Change in distance from top right eyebrow to brim of nose.
            r_n_arc: Change in height from brim of nose to corner of nose.
            r_hl_0: Change in height of left eye from eyelid to eyelid.
            r_hr_0: Change in height of right eye from eyelid to eyelid.
        """
        D_brow = (new[0] - old[0]) / (old[0])  # D
        r_blo = (new[1] - old[1]) / (old[1])  # blo
        r_bli = (new[2] - old[2]) / (old[2])  # bli
        r_bro = (new[3] - old[3]) / (old[3])  # bro
        r_bri = (new[4] - old[4]) / (old[4])  # bri
        r_hl1 = (new[5] - old[5]) / (old[5])  # hl1
        r_hr1 = (new[6] - old[6]) / (old[6])  # hr1
        r_hl2 = - (new[7] - old[7]) / (old[7])  # hl2
        r_hr2 = - (new[8] - old[8]) / (old[8])  # hr2
        r_el = ((new[5] + new[7]) - (old[5] + old[7])) / (old[5] + old[7])  # left eye height
        r_er = ((new[6] + new[8]) - (old[6] + old[8])) / (old[6] + old[8])  # right eye height

        r_h = ((new[23] + new[24]) - (old[23] + old[24])) / (old[23] + old[24])  # lip height
        r_w = (new[25] - old[25]) / old[25]  # lip width
        r_ell = - (new[26] - old[26]) / old[26]  # left lip corner height to nose
        r_r = - (new[27] - old[27]) / old[27]  # right lip corner height to nose
        r_top = - (new[28] - old[28]) / old[28]  # top lip height to nose
        r_btm = - (new[29] - old[29]) / old[29]  # bottom lip height to nose

        return D_brow, r_blo, r_bli, r_bro, r_bri, r_hl1, r_hr1, r_hl2, r_hr2, r_el, r_er, r_h, r_w, r_ell, r_r, r_top, r_btm

    def checkProfile(self, tol):
        """Check that face is looking straight-on at camera.
        
        Check that left jaw is approximately equal to right jaw. Check that distance from eye
        to nose is approximately equal for left and right side.
        
        Args:
            tol (int): Tolerance for how much left side can differ from right side. 
            
        Returns:
            jawBool (bool): True if left and right jaw are the same within tolerance. False otherwise.
            eyeBool (bool):True if left and right jaw are the same within tolerance. False otherwise.
        """
        jawBool = abs(abs(self.nose_line[0, 0] - self.jaw[0, 0]) - abs(self.jaw[-1, 0] - self.nose_line[0, 0])) < tol
        eyeBool = abs(
            abs(self.eye_ell[0, 0] - self.nose_line[0, 0]) - abs(self.eye_r[3, 0] - self.nose_line[0, 0])) < tol

        return jawBool, eyeBool
