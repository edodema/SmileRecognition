"""
NOTE: See what it does
"""

import cv2
from src.detection.landmark import Landmark

lmk = Landmark()

webcam = cv2.VideoCapture(0) 
while True: 
    (_, im) = webcam.read() 
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    points = lmk.detect(img)
    for x, y in points.reshape(points.size//2, 2):
        cv2.circle(im, (x, y), 2, (0, 255, 0), -1)
    
    cv2.imshow('OpenCV', im) 
    if cv2.waitKey(25) & 0xFF == ord('q'): break
    