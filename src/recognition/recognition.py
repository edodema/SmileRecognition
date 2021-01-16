import cv2 

class Recognition():
    def __init__(self, model=cv2.face.LBPHFaceRecognizer_create()):
        """ model should LBPH, Eigen, Fisher """
        self.model = model

    
    
        