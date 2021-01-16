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

lbph = cv2.face.LBPHFaceRecognizer_create()

# Hardcoded :(
start = 10
end = len(os.listdir(files_path))

if start > 0: lbph.read('datasets/lbph.yml')

#lbph.read('datasets/lbph.yml')

for i, file in enumerate(os.listdir(files_path)):
    if start <= i and i < end:
        valences_file = os.path.join(valences_path, re.sub(r"\..*", '.txt', file))
        video_file = os.path.join(files_path, file)

        bboxes, valences, grays = det.get_faces_valence_video(video=video_file, valence_file=valences_file)
        if grays == []: continue
        
        ids = np.array([i for i in range(0, len(grays))])

        if i == 0: 
            lbph.train(grays, ids)
        else:
            lbph.update(grays, ids)
    
lbph.save('datasets/lbph.yml')
