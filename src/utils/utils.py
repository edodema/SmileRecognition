from os.path import split
import cv2
import dlib
import numpy as np
import re
from src.detection.landmark.landmark import Landmark
from src.detection.violajones.violajones import ViolaJones

class Utils:
    def __init__(self): pass
    
    def play(self, path):
        """ Play video """
        cap = cv2.VideoCapture(path)
        if cap.isOpened() == False: print('ERROR! Cannot open video.')
        
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                cv2.imshow('Frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'): break
            else: break
        cap.release()

    def draw_bboxes_video(self, bboxes, path='datasets/affwild/videos/train/105.avi'):
        """ 
        Draw boundary boxes on a video
        path: path of the video to play
        bboxes: the whole list of boundary boxes frame per frame
        NOTE: It is just too bad, is better to detect them with Viola Jones 
        """
        cap = cv2.VideoCapture(path)
        if cap.isOpened() == False: print('ERROR! Cannot open video.')
        
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:

                index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                print(index)
                if faces := bboxes.get(index):
                    for face in faces: cv2.rectangle(frame, tuple(face[0]), tuple(face[1]), (0,255,0), 2)
                else: pass

                cv2.imshow('Frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'): break

            else: break
        cap.release()

    def draw_landmarks_video(self, video_path, landmark_detector, img_size=224):
        """ 
        Draw landmark on a video
        video: video path
        landmark_detector: 
        """
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened() == False: print('ERROR! Cannot open video.')
        
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                landmark_detector.draw(frame)

                cv2.imshow('Frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'): break

            else: break
        cap.release()
        
    def get_valences_landmarks_video(self, video_path, valence_file_path, feature_detector, k=0.5):
        """
        Get landmarks and correspodning 
        valences for each frame in a video
        """
        ret_valences, ret_features = np.empty((0,), dtype=np.uint8), np.empty((0, len(feature_detector.points*2)), dtype=np.uint32)

        cap = cv2.VideoCapture(video_path)
        valences = np.loadtxt(valence_file_path)

        if cap.isOpened() == False: print('ERROR! Cannot open video.')
        
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                features = feature_detector.extract_features_img(frame)
                """ 
                Get only valences from frames s.t.
                1. We have detected a face 
                2. -k <= valence <= k
                """
                index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if index >= len(valences): continue # don't completely get why, is in the dataset

                valence = valences[index]
                if features.shape[0] > 0 and not( -k <= valence and valence <= k):
                    """ 
                    Get the largest landmark measured as the euclidean distance
                    between the first and last point
                    """
                    feature = features[0] if features.shape[0] == 1 else features[np.where((features[:,0]-features[:,-2])*(features[:,0]-features[:,-2]) + (features[:,1]-features[:,-1])*(features[:,1]-features[:,-1]) == max(list(map(lambda feature: (feature[0]-feature[-2])*(feature[0]-feature[-2]) +  (feature[1]-feature[-1])*(feature[1]-feature[-1]), features))))[0]][0]

                    ret_features = np.append(ret_features, [feature], axis=0)
                    ret_valences = np.append(ret_valences, [0 if valence < 0 else 1])
                    
            else: break

        cap.release()
        return ret_valences, ret_features
