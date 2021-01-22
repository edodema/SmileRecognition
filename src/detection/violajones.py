import cv2

class ViolaJones():
    """
    Implements Viola Jones detection with Haar features.

    Attributes
    ----------
    haar: Path of the Haar features' file.
    """
    def __init__(self, haar ='datasets/haar/haarcascade_frontalface_default.xml'):
        self.cascade = cv2.CascadeClassifier(haar)
        
        
    def detect(self, img):
        """
        Detect faces.

        Input
        -----
        img: Image as a 2D array where to search faces for.

        Output
        ------
        bboxes: Bounding boxes of detected faces.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bboxes = self.cascade.detectMultiScale(gray, 1.3, 5)
        return bboxes
           