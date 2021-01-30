""" 
Get landmark features and valences, by default the whole directory is explored.
"""

import os, re, fire
import numpy as np
from src.detection.landmark import Landmark
from src.utils.utils import Utils

def main(training_perc=1, videos_dir='datasets/affwild/videos/train', valences_dir='datasets/affwild/annotations/train/valence',
features_output='datasets/features_landmark.txt', valences_output='datasets/valences_landmark.txt'):
    assert 0 <= training_perc and training_perc <= 1, "The percentage of the training set size must be between 0 and 1."

    utils = Utils()
    landmark = Landmark()
    
    dataset_size = len(os.listdir(videos_dir))
    end = int(training_perc * dataset_size)

    valences_tot = np.empty((0,), dtype=np.uint8)
    features_tot = np.empty((0,len(landmark.points)*2), dtype=np.uint)

    for i, file in enumerate(os.listdir(videos_dir)):
        if i < end:
            valences_file = os.path.join(valences_dir, re.sub(r"\..*", '.txt', file))
            video_file = os.path.join(videos_dir, file)
            valences, features = utils.get_valences_landmarks_video(video_file, valences_file, landmark)
            valences_tot = np.append(valences_tot, valences)
            features_tot = np.append(features_tot, features, axis=0)

    np.savetxt(features_output, features_tot)
    np.savetxt(valences_output, valences_tot)

if __name__=='__main__':
    fire.Fire(main)
