import numpy as np
import os
from numpy.core.fromnumeric import sort
import pandas as pd
import re
from src.utils.utils import Utils

class RefineDataset(Utils):
    """ Class used to refine the First Affect-in-the-Wild Challenge  dataset for our purposes """
    def __init__(self, path):
        super().__init__()

    def refine(self, files_path='datasets/affwild/videos/train', 
    bboxes_path='datasets/affwild/bboxes/train', 
    valences_path='datasets/affwild/annotations/train/valence',
    csv_path='datasets/data.csv'):
        """
        Build up a dataframe with columns 
        video | frame | bbox | valence 
        """ 
        arr = []
        for file in os.listdir(files_path):
            bboxes_dir = os.path.join(bboxes_path, re.sub(r"\..*", '', file))
            valences_file = os.path.join(valences_path, re.sub(r"\..*", '.txt', file))

            valences = np.loadtxt(valences_file)

            # Array from which the dataframe will be built

            for bboxes_file in os.listdir(bboxes_dir):
                bboxes_file_path = os.path.join(bboxes_dir, bboxes_file)
                # The frame number is took directly from the bbox file's name
                frame = int(re.sub(r"\..*", '', bboxes_file))

                """ 
                NOTE:
                Strangely this is a problem, don't know why there is 
                one bbox exceeding. I ignore it
                """ 
                if frame < len(valences): 
                    valence = valences[frame]

                    if not( -0.1 <= valence and valence <= 0.1): # to avoid  only noise
                        bboxes = self.get_bboxes(bboxes_file_path)
                        rectangles = [ [tuple(bbox[0]), tuple(bbox[2])] for bbox in bboxes ]
                    
                        arr += [[file, frame, rectangles, valence]]

        df = pd.DataFrame(np.array(arr, dtype=object), columns=['video', 'frame', 'faces', 'valence'])
        df.to_csv(csv_path)
                
    def load_data(self):
        """ used to load the saved csv in a useful way """
        pass
    