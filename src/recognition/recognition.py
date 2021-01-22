import cv2
import numpy as np

class RecognitionSVM():
    """
    Emotion recognition through a SVM classifier.
    """
    def __init__(self): pass

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
        
        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
        svm.train(samples, cv2.ml.ROW_SAMPLE, responses)

        svm.save(output_path)

    def predict(self, svm, landmark):
        """
        Predict the value of a landmark encoding.

        Input
        -----
        svm: SVM object for detection.
        landmark: Landmark array for which the SVM will give a prediction.

        Output
        ------
        prediction: The prediction of the SVM.
        """
        test = landmark.reshape((1, len(landmark)))
        
        # NOTE: First the SVM needs to be trained
        prediction = int(svm.predict(test)[1][0,0])
        return prediction