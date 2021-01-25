import cv2, fire
from src.detection.violajones import ViolaJones
from src.detection.landmark import Landmark
from src.utils.utils import Utils
from src.recognition.recognition import Recognition

class Demo:
    """
    Online demo.
    """
    
    def __init__(self, utils=Utils(), violajones=ViolaJones(), landmark=Landmark(), recognition=Recognition()):
        self.utils = utils
        self.violajones = violajones
        self.landmark = landmark
        self.recognition = recognition

    def exec(self, svm_path):
        """
        Execute the online demo.

        Input
        -----
        svm_path: Path of the SVM trained model.
        """
        svm = self.recognition.load(svm_path)

        webcam = cv2.VideoCapture(0)
        
        while True: 
            (_, im) = webcam.read() 
            img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            
            bboxes = self.violajones.detect(im)
            if len(bboxes) > 0:
                lmk = self.landmark.detect(img)
                if lmk.shape[0] > 0:
                    value = self.recognition.predict(svm, lmk)
                    color = (0, 255, 0) if value == 1 else (0, 0, 255)
                    
                    #for x, y in lmk.reshape(lmk.size//2, 2): cv2.circle(im, (x, y), 2, color, -1) # Draw landmarks
                    for x, y, w, h in bboxes: cv2.rectangle(im, (x, y), (x+w, y+h), color, 2) # Draw bbox
            
            cv2.imshow('OpenCV', cv2.flip(im, 1))
            if cv2.waitKey(25) & 0xFF == ord('q'): break

if __name__ == '__main__':
    svm_path = 'datasets/svm_complete.yml'

    demo = Demo()
    demo.exec(svm_path)