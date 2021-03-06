import cv2
import numpy as np
from sklearn import svm
import joblib

class SVM():
    """
    Emotion recognition through a SVM classifier.
    """
    def __init__(self, type, kernel='0', degree=1):
        """
        Instantiate a SVM

        input
        -----
        type: The tyoe of SVM, can be cv2 if the OpenCV or skl if Sklearn type.
        """
        assert type == 'cv2' or type == 'skl', "SVM type has to be 'cv2' or 'skl'."

        if kernel == '0': assert degree == 1 , "If the function is linear the degree should be 1."
        elif kernel == '1': assert degree > 1,  "A polynomial function needs degree higher than 0."

        kernels = {'0':cv2.ml.SVM_LINEAR, '1':cv2.ml.SVM_POLY, '2':cv2.ml.SVM_RBF}
        self.__type = type
        self.__kernel = kernels[kernel]
        self.__degree = degree

    def train_svm(self, samples_path, responses_path, output_path):
        """
        Train an SVM classifier.

        Input
        -----
        samples_path: Path of samples/data to train the SVM with.
        responses_path: Path of responses/labels to train the SVM with.
        output_path: Path of the output file where the SVM will be saved.

        NOTE: samples are landmarks and responses are valences.
        """
        samples = np.loadtxt(samples_path, dtype=np.float32)
        responses = np.loadtxt(responses_path).astype(int)
        
        if self.__type == 'cv2':
            # The SVM is a OpenCV one
            svm_cv2 = cv2.ml.SVM_create()
            svm_cv2.setType(cv2.ml.SVM_C_SVC)
            svm_cv2.setKernel(self.__kernel)
            svm_cv2.setDegree(self.__degree)
            svm_cv2.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
            svm_cv2.train(samples, cv2.ml.ROW_SAMPLE, responses)
            svm_cv2.save(output_path)

        elif self.__type == 'skl': 
            # The SVM is a Sklearn one
            svm_skl = svm.SVC()
            svm_skl.fit(samples, responses)
            joblib.dump(svm_skl, output_path)
        
        else: 
            print("ERROR! Wait This should not happen.")
            exit

    def predict(self, svm, test):
        """
        Subroutine to predict the value of a landmark encoding.
        NOTE: First the SVM needs to be trained


        Input
        -----
        svm: SVM object for detection.
        test: Landmark array for which the SVM will give a prediction, .

        Output
        ------
        prediction: The prediction of the SVM.

        TODO: This can be the same for both cv2 and sklearn
        """
        prediction = svm.predict(test)
        return prediction

    def load(self, svm_path):
        """
        Predict the value of a landmark encoding.

        Input
        -----
        svm_path: Path for the SVM object for detection.

        Output
        ------
        svm: The loaded SVM object.
        """
        svm = None
        
        if self.__type == 'cv2': 
            # The SVM is a OpenCV one
            svm = cv2.ml.SVM_create().load(svm_path)

        elif self.__type == 'skl': 
            # The SVM is a Sklearn one
            svm = joblib.load(svm_path)

        else: 
            print("ERROR! Wait This should not happen.")
            exit

        return svm
