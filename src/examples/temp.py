from src.detection.violajones.violajones import ViolaJones
from src.detection.landmark.landmark import Landmark
from src.utils.utils import Utils

landmark = Landmark()
ut = Utils()

video = 'datasets/affwild/videos/train/318.mp4'

files_path='datasets/affwild/videos/train' 
bboxes_path='datasets/affwild/bboxes/train'
valence_file_path='datasets/affwild/annotations/train/valence/318.txt'

#ut.draw_landmarks_video(video, landmark)
ut.get_valences_landmarks_video(video, valence_file_path, Landmark())