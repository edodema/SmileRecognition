import os
import cv2
import re
import numpy as np
from src.utils.refine_dataset import RefineDataset
from src.utils.utils import Utils
from src.recognition.recognition import Recognition
from src.detection.detection import Detection

det = Detection()
files_path='datasets/affwild/videos/train' 
bboxes_path='datasets/affwild/bboxes/train'
valences_path='datasets/affwild/annotations/train/valence'

grays_tot = []

for i, file in enumerate(os.listdir(files_path)):
    valences_file = os.path.join(valences_path, re.sub(r"\..*", '.txt', file))
    video_file = os.path.join(files_path, file)

    bboxes, valences, grays = det.get_faces_valence_video(video=video_file, valence_file=valences_file)
    grays_tot += grays

if grays_tot == []: 
    print("ERROR! There is no data to train.")

lbph = cv2.face.LBPHFaceRecognizer_create()
eigen = cv2.face.EigenFaceRecognizer_create()
fisher = cv2.face.FisherFaceRecognizer_create()

ids = np.array([i for i in range(0, len(grays_tot))])
        
lbph.train(grays_tot, ids)
eigen.train(grays_tot, ids)
fisher.train(grays_tot, ids)
    
lbph.save('datasets/lbph.yml')
eigen.save('datasets/eigen.yml')
fisher.save('datasets/fisher.yml')
