import os
import cv2
import re
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
from src.utils.refine_dataset import RefineDataset
from src.utils.utils import Utils
from src.recognition.recognition import Recognition
from src.detection.detection import Detection
import matplotlib.pyplot as plt

det = Detection()
files_path='datasets/affwild/videos/train' 
bboxes_path='datasets/affwild/bboxes/train'
valences_path='datasets/affwild/annotations/train/valence'

# Hardcoded :(
dataset_size = len(os.listdir(files_path))
training_perc = 0.05
start = 0
end = 10 #int(training_perc * dataset_size)

radius=1
numPoints = 8
eps = 1e-7

"""
label = np.empty((0,), dtype=np.uint8)
data = np.empty((0,10), dtype=np.float64)

for i, file in enumerate(os.listdir(files_path)):
    if start <= i and i < end:
        print(file)
        valences_file = os.path.join(valences_path, re.sub(r"\..*", '.txt', file))
        video_file = os.path.join(files_path, file)

        valences, grays = det.get_faces_valence_video(video=video_file, valence_file=valences_file)
        label = np.append(label, valences)
        ids = np.array([i for i in range(0, len(grays))])

        if grays.size > 0:
            for gray in grays:
                lbp = local_binary_pattern(gray, numPoints, radius, method="uniform")
                (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3),range=(0, numPoints + 2))            
                hist = hist.astype("float")                         
                hist /= (hist.sum() + eps)
                data = np.append(data, [hist],axis=0)

np.savetxt('labels.txt', label)
np.savetxt('data.txt', data)

"""

label = np.loadtxt('labels.txt')
data = np.loadtxt('data.txt')

model = LinearSVC(C=100.0, random_state=42)
model.fit(data, label)

#prediction = model.predict(data[0])[0]
#print(prediction)