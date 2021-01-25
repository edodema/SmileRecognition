"""
Temporary file that can be removed.
NOTE: This file is like a sandbox, use all this clutter file to get something good in a test class.
NOTE: Probably can be deleted.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import svm
from src.detection.violajones import ViolaJones
from src.detection.landmark import Landmark
from src.utils.utils import Utils
from src.recognition.recognition import Recognition
from src.test.test import Test

def plot_roc_cur(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


video = 'datasets/affwild/videos/train/318.mp4'
files_path='datasets/affwild/videos/train' 
bboxes_path='datasets/affwild/bboxes/train'
valence_file_path='datasets/affwild/annotations/train/valence/318.txt'
landmarks_path = 'datasets/features_landmark_complete.txt'
valences_path = 'datasets/valences_landmark_complete.txt'
svm_path = 'datasets/svm_complete.yml'

landmark = Landmark()
violajones = ViolaJones()
ut = Utils()
recognition = Recognition()
test = Test(landmarks_path, valences_path)

X_train, X_test, y_train, y_test = test.split_dataset()

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

clf = svm.SVC()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
"""
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_POLY)
svm.setDegree(3)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

test = np.arange(64).astype(np.float32).reshape(1,64)

prediction = svm.predict(X_test)[1].flatten().astype(int)
"""

fper, tper, thresholds = metrics.roc_curve(y_test, prediction) 
auc = metrics.auc(fper, tper)
print(auc)
plot_roc_cur(fper, tper)