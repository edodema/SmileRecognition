import cv2 
import dlib 
import numpy as np

class Landmark():
    def __init__(self, path='datasets/landmarks.dat', points=[ i for i in range(36, 68)]):
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_detector = dlib.shape_predictor(path)
        self.points = points


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
        
    def extract_features(self, face, rectangle):
        """
        Extract features from a face
        """
        landmark_points = np.empty((0,), dtype=np.int)
        landmarks = self.landmark_detector(face, rectangle)
        for n in self.points:
            point = landmarks.part(n)
            landmark_points = np.append(landmark_points, point.x)
            landmark_points = np.append(landmark_points, point.y)
        return landmark_points

    def extract_features_img(self, img):
        """
        Extract features from a face
        """
        features = np.empty((0, len(self.points*2)), dtype=np.uint32)
        faces = self.face_detector(img, 1)
        for _, rectangle in enumerate(faces):
            landmark = self.extract_features(img, rectangle)
            features = np.append(features, [landmark], axis=0)
        return features

    def draw(self, img):
        """
        Draw landmark points on an image
        """
        faces = self.face_detector(img,1)
        for _, rectangle in enumerate(faces):
            landmarks = self.extract_features(img, rectangle)[:len(self.points)*2].reshape(len(self.points), 2)
            for point in landmarks: cv2.circle(img, (point[0], point[1]), 2, (0, 255, 0), -1)
