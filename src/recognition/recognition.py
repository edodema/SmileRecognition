import cv2 

class Recognition():
    """
    NOTE: Remove
    """
    def __init__(self, model=cv2.face.LBPHFaceRecognizer_create()):
        """ model should LBPH, Eigen, Fisher """
        self.model = model

        
    
        