import pytesseract as py
import cv2
import numpy as np



def image_ocr(file):
    return py.image_to_string(file)

def image_to_gray(file):
    return cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)

def image_blurrer(file):
    return cv2.GaussianBlur(file, (9,9),0)

def image_thresher(file):
    return cv2.threshold(file, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

def image_dilate(file):
    return cv2.dilate(file, cv2.getStructuringElement(cv2.MORPH_RECT, (30,5)), iterations=5)

def image_skew_angle(file):
    contours, hierarchy = cv2.findContours(np.float32(file), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    angle = cv2.minAreaRect(contours[0])[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

def image_rotate(file, angle):
    copy = file.copy()
    (h,w) = copy.shape[:2]
    center = (w//2, h//2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(copy, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
def image_deskewer(file):
    return image_rotate(file, -1.0 * image_skew_angle(file))
