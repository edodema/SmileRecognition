import cv2, fire
import numpy as np
from src.detection.violajones import ViolaJones
from src.detection.landmark import Landmark
from src.recognition.svm import SVM

class Demo:
    """
    Online demo.
    NOTE: Change SVM with the sklearn one.
    """
    
    def __init__(self, recognition, violajones=ViolaJones(), landmark=Landmark()):
        self.__violajones = violajones
        self.__landmark = landmark
        self.__recognition = recognition

    def exec(self, svm_path):
        """
        Execute the online demo.

        Input
        -----
        svm_path: Path of the SVM trained model.
        """
        svm = self.__recognition.load(svm_path)

        webcam = cv2.VideoCapture(0)
        
        while True: 
            (_, im) = webcam.read() 
            img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            
            bboxes = self.__violajones.detect(im)
            if len(bboxes) > 0:
                test = self.__landmark.detect(img)
                if test.shape[0] > 0: 
                    value = self.__recognition.predict(svm, test.reshape(1, 64).astype(np.float32))[0]
                    color = (0, 255, 0) if value == 1 else (0, 0, 255)
                    
                    for x, y in test.reshape(test.size//2, 2): cv2.circle(im, (x, y), 2, color, -1) # Draw landmarks
                    for x, y, w, h in bboxes: cv2.rectangle(im, (x, y), (x+w, y+h), color, 2) # Draw bbox
            
            cv2.imshow('OpenCV', cv2.flip(im, 1))
            if cv2.waitKey(25) & 0xFF == ord('q'): break

if __name__ == '__main__':
    svm_path = 'datasets/svm_complete.yml'

    recognition = SVM('skl')

    demo = Demo(recognition)
    demo.exec(svm_path)
