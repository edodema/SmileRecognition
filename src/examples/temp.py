"""
Temporary file that can be removed.
NOTE:This file is like a sandbox, use all this clutter file to get something good in a test class.
"""

import numpy as np
import cv2
from src.detection.violajones import ViolaJones
from src.detection.landmark import Landmark
from src.utils.utils import Utils
from src.recognition.recognition import Recognition

landmark = Landmark()
violajones = ViolaJones()
ut = Utils()
recognition = Recognition()

video = 'datasets/affwild/videos/train/318.mp4'

files_path='datasets/affwild/videos/train' 
bboxes_path='datasets/affwild/bboxes/train'
valence_file_path='datasets/affwild/annotations/train/valence/318.txt'
landmarks_path = 'datasets/features_landmark_complete.txt'
valences_path = 'datasets/valences_landmark_complete.txt'
svm_path = 'datasets/svm_complete.yml'

#ut.play(video)
#ut.draw_landmarks_video(video, landmark)
#ut.draw_bboxes_video(video, violajones)
#ut.online_violajones(violajones)

#recognition.train_svm(landmarks_path, valences_path, svm_path)

svm = recognition.load(svm_path)

# Online recognition
webcam = cv2.VideoCapture(0)
while True: 
    (_, im) = webcam.read() 
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    bboxes = violajones.detect(im)
    if len(bboxes) > 0:
        lmk = landmark.detect(img)
        if lmk.shape[0] > 0:
            value = recognition.predict(svm, lmk)
            color = (0, 255, 0) if value == 1 else (0, 0, 255)

            #for x, y in lmk.reshape(lmk.size//2, 2): cv2.circle(im, (x, y), 2, color, -1)
            for x, y, w, h in bboxes: cv2.rectangle(im, (x, y), (x+w, y+h), color, 2)


    cv2.imshow('OpenCV', im) 
    if cv2.waitKey(25) & 0xFF == ord('q'): break