"""
Predict with LBP

NOTE: Remove it since I decided to use landmarks and svm.
"""

import cv2
import numpy as np

files_path='datasets/affwild/videos/train'
haar = 'datasets/haar/haarcascade_frontalface_default.xml'
valences_path = 'datasets/valences.txt'

face_cascade = cv2.CascadeClassifier(haar) 
webcam = cv2.VideoCapture(0) 
img_size = 224

print("Start Loading")
lbph = cv2.face.LBPHFaceRecognizer_create()
lbph.read('datasets/lbph_online.yml')
valences = np.loadtxt(valences_path, dtype=np.uint8)
print(valences)
print("Loading ended") 

while True: 
    (_, im) = webcam.read() 
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    for (x, y, w, h) in faces: 
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
        face = cv2.resize(gray[y:y + h, x:x + w], (img_size, img_size))

        id, score = lbph.predict(face)
        
        if score <500: 
            valence = valences[id]
            emotion = 'Happy' if valence == 1 else 'Sad'
            cv2.putText(im, emotion, (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        else: 
            cv2.putText(im, 'Not recognized', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0)) 
    cv2.imshow('OpenCV', im) 
    if cv2.waitKey(25) & 0xFF == ord('q'): break