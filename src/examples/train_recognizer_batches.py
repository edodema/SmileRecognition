import os
import cv2
import re
import numpy as np
from src.utils.refine_dataset import RefineDataset
from src.utils.utils import Utils
from src.recognition.recognition import Recognition
from src.detection.detection import Detection
import matplotlib.pyplot as plt

det = Detection()
files_path='datasets/affwild/videos/train' 
bboxes_path='datasets/affwild/bboxes/train'
valences_path='datasets/affwild/annotations/train/valence'

lbph = cv2.face.LBPHFaceRecognizer_create()

# Hardcoded :(
dataset_size = len(os.listdir(files_path))
training_perc = 0.10 
start = 0
end = int(training_perc * dataset_size)

valences_tot = np.empty((0,), dtype=np.uint8)

if start > 0:
    valences_tot = np.loadtxt('datasets/valences.txt')
    lbph.read('datasets/lbph.yml') 

for i, file in enumerate(os.listdir(files_path)):
    if start <= i and i < end:
        valences_file = os.path.join(valences_path, re.sub(r"\..*", '.txt', file))
        video_file = os.path.join(files_path, file)

        valences, grays = det.get_faces_valence_video(video=video_file, valence_file=valences_file)
        valences_tot = np.append(valences_tot, valences)
        ids = np.array([i for i in range(0, len(grays))])

        if grays.size > 0:
            if i == 0 : lbph.train(grays, ids)
            else: lbph.update(grays, ids)

np.savetxt('datasets/valences.txt', valences_tot)    
lbph.save('datasets/lbph.yml')
