import cv2 
import dlib 
import numpy as np

class Landmark():
    def __init__(self, path='datasets/landmarks.dat', feature_points=[ i for i in range(0, 68)]):
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_detector = dlib.shape_predictor(path)
        self.points = feature_points


    def detect(self, img):
        """ 
        Detect landmark points 
        """
        faces = self.face_detector(img, 1)
        landmark_points = np.empty((0,), dtype=np.int)
        
        for _, d in enumerate(faces):
            landmarks = self.landmark_detector(img, d)
            for n in self.points:
                point = landmarks.part(n)
                landmark_points = np.append(landmark_points, point.x)
                landmark_points = np.append(landmark_points, point.y)
        return landmark_points
