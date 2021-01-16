import cv2
import numpy as np

class Detection:
    def __init__(self): pass

    def detect_faces_video(self, 
    path='datasets/affwild/videos/train/105.avi', 
    haar='datasets/haar/haarcascade_frontalface_default.xml'):
        """
        Detect faces.

        path: path of the video.
        haar: path of the haar feature classifier. 
        """
        face_cascade = cv2.CascadeClassifier(haar)
        cap = cv2.VideoCapture(path)
        
        if cap.isOpened() == False: print('ERROR! Cannot open video.')
        
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

                cv2.imshow('Frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'): break

            else: break
        cap.release()
    
    def get_faces_valence_video(self, 
    video='datasets/affwild/videos/train/105.avi', 
    haar ='datasets/haar/haarcascade_frontalface_default.xml',
    valence_file = 'datasets/affwild/annotations/train/valence/105.txt'):
        """
        Detect faces.

        path: path of the video.
        haar: path of the haar feature classifier.
        valence: path of the valence file
        """
        ret_bboxes, ret_valences, ret_gray = [], [], []

        face_cascade = cv2.CascadeClassifier(haar)
        cap = cv2.VideoCapture(video)

        valences = np.loadtxt(valence_file)

        if cap.isOpened() == False: print('ERROR! Cannot open video.')
        
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                # Gray out images
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                """ 
                Get only valences from frames s.t.
                1. We have detected a face 
                2. -k <= valence <= k 
                """
                k = 0.1
                index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if index >= len(valences): continue
                valence = valences[index]

                if len(faces) > 0 and not( -k <= valence and valence <= k):
                    ret_valences += [0 if valence < 0 else 1]
                    # Get bbox with largest area
                    bbox = faces[0] if len(faces) == 1 else faces[np.where(faces[:,2]*faces[:,3] == max(list(map(lambda r: r[2]*r[3], faces))))[0]][0]
                    ret_bboxes += [bbox]
                    # Save gray images for the recognizer
                    x, y, w, h = bbox 
                    img = cv2.resize(frame[y:y+h, x:x+w], (224,224))
                    ret_gray += [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)]  
    
            else: break
        cap.release()
        return ret_bboxes, ret_valences, ret_gray
