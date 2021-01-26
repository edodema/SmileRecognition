from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

class Test:
    """
    Implements tests for the datasets.
    """
    def __init__(self, samples_path, labels_path): 
        self.__X = np.loadtxt(samples_path, dtype=np.float32)
        self.__y = np.loadtxt(labels_path).astype(int)
        self.__X_train = []
        self.__X_test = []
        self.__y_train = []
        self.__y_test = []

    def split_dataset(self, test_size=0.2, random_state=42):
        """
        Split dataset in train and test.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.__X, self.__y, test_size=test_size, random_state=random_state)

        self.__X_train = X_train
        self.__X_test = X_test
        self.__y_train = y_train
        self.__y_test = y_test

        return X_train, X_test, y_train, y_test

    def plot_roc(self, fpr, tpr):
        """
        Plot ROC curve

        input
        -----
        fpr: False positive rate of predictions.
        tpr: True positive rate of predictions.

        NOTE: https://www.codespeedy.com/how-to-plot-roc-curve-using-sklearn-library-in-python/
        """  
        plt.plot(fpr, tpr, color='orange', label='ROC')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()

    
    def metric_roc(self, prediction):
        """
        Plot ROC curve and measure AUROC

        input
        -----
        prediction: A list of test data.

        output
        ------
        fpr: False positive rate of predictions.
        tpr: True positive rate of predictions.
        auc: AUC.
        ths: Thresholds.
        """
        fpr, tpr, ths = metrics.roc_curve(self.__y_test, prediction) 
        auc = metrics.auc(fpr, tpr)
        return fpr, tpr, auc, ths
