import cv2
import numpy as np

class Utils:
    """
    Support functions.
    """
    def __init__(self): pass
    
    def play(self, path):
        """ 
        Play a video 
        
        Input
        -----
        path = Video path.
        """
        cap = cv2.VideoCapture(path)
        if cap.isOpened() == False: print('ERROR! Cannot open video.')
        
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                cv2.imshow('Frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'): break
            else: break
        cap.release()

    def draw_bboxes_video(self, video_path, violajones_detector):
        """ 
        Given the boundary boxes draw them on video.

        Input
        -----
        path: Path of the video to play.
        haar: Haar features to detect. 
        """
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened() == False: print('ERROR! Cannot open video.')
        
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:

                bboxes = violajones_detector.detect(frame)                
                for x, y, w, h in bboxes: cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

                cv2.imshow('Frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'): break

            else: break
        cap.release()

    def draw_landmarks_video(self, video_path, landmark_detector):
        """ 
        Draw landmarks on a video.

        Input
        -----
        video_path: Path of the video.
        landmark_detector: Landmark detector object.
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
        Get landmarks and corresponding valences for each frame in a video.

        Input
        -----
        video_path: The path of the video.
        valence_file_path: Path of the file containing the valences.
        feature_detector: Feature detector object.
        k: Threshold of valences' values, ones between -k and k will be ignored.

        Output
        ------
        ret_valences: Array with filtered valences.
        ret_features: Array with filtered landmark features.
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

    def online_violajones(self, violajones_detector):
        """
        Open webcam and see Viola Jones detection.
        """
        webcam = cv2.VideoCapture(0)
        while True: 
            (_, im) = webcam.read() 
            img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            bboxes = violajones_detector.detect(im)   
            for x, y, w, h in bboxes: cv2.rectangle(im, (x, y), (x+w, y+h), (0,255,0), 2)
            
            cv2.imshow('OpenCV', im) 
            if cv2.waitKey(25) & 0xFF == ord('q'): break

    def online_landmark(self, landmark_detector):
        """
        Open webcam and see landmark detection.
        """
        webcam = cv2.VideoCapture(0) 
        while True: 
            (_, im) = webcam.read() 
            img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            points = landmark_detector.detect(img)
            for x, y in points.reshape(points.size//2, 2): cv2.circle(im, (x, y), 2, (0, 255, 0), -1)
            
            cv2.imshow('OpenCV', im) 
            if cv2.waitKey(25) & 0xFF == ord('q'): break
