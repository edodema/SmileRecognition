"""
Predict svm for landmark features.
"""

import os
import numpy as np
import cv2
from numpy.core.fromnumeric import shape
from src.detection.landmark import Landmark

"""

det = Landmark()

responses = np.loadtxt('datasets/valences_landmark.txt').astype(int)
samples = np.loadtxt('datasets/features_landmark.txt', dtype=np.float32)

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
svm.train(samples, cv2.ml.ROW_SAMPLE, responses)

test = np.arange(len(samples[0])).reshape(1, len(samples[0])).astype(np.float32)

prediction = int(svm.predict(test)[1][0,0])

out = 'Happy' if prediction == 1 else 'Sad'

# NOTE: Check 1 -1 in svm 

print(out)

svm.save('datasets/svm_landmark.yml')
"""
samples = np.loadtxt('datasets/features_landmark.txt', dtype=np.float32)
test = np.arange(len(samples[0])).reshape(1, len(samples[0])).astype(np.float32)

svm = cv2.ml.SVM_create()
svm = svm.load('datasets/svm_landmark.yml')
prediction = int(svm.predict(test)[1][0,0])
out = 'Happy' if prediction == 1 else 'Sad'

print(out)