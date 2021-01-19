import os
import re
import numpy as np
from src.detection.violajones.violajones import ViolaJones
from src.detection.landmark.landmark import Landmark
from src.utils.utils import Utils

img_size = 64

files_path='datasets/affwild/videos/train' 
bboxes_path='datasets/affwild/bboxes/train'
valences_path='datasets/affwild/annotations/train/valence'

utils = Utils()
violajones = ViolaJones()
landmark = Landmark()


# Hardcoded :(
dataset_size = len(os.listdir(files_path))
training_perc = 1
end = int(training_perc * dataset_size)

valences_tot = np.empty((0,), dtype=np.uint8)
features_tot = np.empty((0,len(landmark.points)*2), dtype=np.uint)

for i, file in enumerate(os.listdir(files_path)):
    if i < end:
        valences_file = os.path.join(valences_path, re.sub(r"\..*", '.txt', file))
        video_file = os.path.join(files_path, file)
        valences, features = utils.get_valences_landmarks_video(video_file, valences_file, landmark)
        valences_tot = np.append(valences_tot, valences)
        features_tot = np.append(features_tot, features, axis=0)

np.savetxt('datasets/valences_landmark_complete.txt', valences_tot)
np.savetxt('datasets/features_landmark_complete.txt', features_tot)
