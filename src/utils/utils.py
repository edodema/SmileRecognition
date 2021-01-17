from os.path import split
import cv2
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
        
    def get_valences_landmarks_video(self, video_path, valences_path, frame_size, face_detector, feature_detector, k=0.5):
        """
        Get landmarks and correspodning 
        valences for each frame in a video
        """
        ret_valences, ret_features = np.empty((0,), dtype=np.uint8), np.empty((0, len(feature_detector.points*2)), dtype=np.uint32)

        cap = cv2.VideoCapture(video_path)
        valences = np.loadtxt(valences_path)

        if cap.isOpened() == False: print('ERROR! Cannot open video.')
        
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                bboxes = face_detector.detect(frame, frame_size)
                """ 
                Get only valences from frames s.t.
                1. We have detected a face 
                2. -k <= valence <= k
                """
                index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if index >= len(valences): continue # don't completely get why, is in the dataset

                valence = valences[index]
                if len(bboxes) > 0 and not( -k <= valence and valence <= k):
                    # Get bbox with largest area
                    x, y, w, h = bboxes[0] if len(bboxes) == 1 else bboxes[np.where(bboxes[:,2]*bboxes[:,3] == max(list(map(lambda r: r[2]*r[3], bboxes))))[0]][0]
                    # Seath landmark features in a face
                    img = frame[y:y+h, x:x+w]
                    features = feature_detector.detect(img)
                    # Check if features have been found
                    if len(features) > 0:
                        ret_features = np.append(ret_features, [features], axis=0)
                        ret_valences = np.append(ret_valences, [0 if valence < 0 else 1])
                    
            else: break

        cap.release()
        return ret_valences, ret_features

