"""
Train svm classifier with landmark features.
"""

import cv2
import numpy as np
from numpy.lib import histograms
import pandas as pd

model_path = 'datasets/lbph.yml'
valences_path = 'datasets/valences.txt'

model = cv2.face.LBPHFaceRecognizer_create()
model.read(model_path)

histograms = np.array(model.getHistograms())

"""
Histograms are the training data
Valences are the labels
"""
samples = np.array(histograms[:,0,:])
responses = np.loadtxt(valences_path).astype(int)

# Train the SVM
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
svm.train(samples, cv2.ml.ROW_SAMPLE, responses)

svm.save('datasets/svm.yml')
