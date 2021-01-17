import re
import cv2
import numpy as np

class ViolaJones():
    def __init__(self, haar ='datasets/haar/haarcascade_frontalface_default.xml'):
        self.cascade = cv2.CascadeClassifier(haar)
        
        
    def detect(self, img, img_size):
        ret_img = np.empty((0, img_size, img_size, 3), dtype=np.uint8) 

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bboxes = self.cascade.detectMultiScale(gray, 1.3, 5)
        """
        if len(bboxes) > 0:
            for x, y, w, h in bboxes:
                cropped = cv2.resize(img[y:y+h, x:x+w], (img_size, img_size))
                ret_img = np.append(ret_img, [cropped], axis=0)
        """
        return bboxes
                    