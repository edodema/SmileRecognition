from os.path import split
import cv2
import numpy as np
import re

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

    def get_bboxes(self, path):
        """ Get bboxes from a pts file """
        f = open(path, 'r+')
        bboxes = f.read()
        f.close()
        """ 
        Extract possible bboxes, I assume
        there can be multiple {...}{...}
        """
        bboxes = re.sub(r"\n", ',', bboxes)
        bboxes = re.findall(r"{.*?}", bboxes)
        bboxes = [ [ [ int(float(x)) for x in el.split() ] for el in list(line[2:-2].split(',')) ] for line in bboxes ] 

        return np.array(bboxes)

    def draw_bboxes_video(self, path, bboxes):
        """ 
        Draw boundary boxes on a video

        path: path of the video to play
        bboxes: the whole list of boundary boxes frame per frame

        TODO: fix drawing caused by array nesting 
        """
        cap = cv2.VideoCapture(path)
        if cap.isOpened() == False: print('ERROR! Cannot open video.')
        
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:

                index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                print(index)
                if bboxes.get(index):
                    x,y = bboxes[index]
                    cv2.rectangle(frame, x, y, (0,255,0), 2)
                else: pass

                cv2.imshow('Frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'): break

            else: break
        cap.release()
