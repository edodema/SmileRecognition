from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.recognition.recognition import Recognition

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

if __name__ == '__main__':
    samples_path = 'datasets/features_landmark_complete.txt'
    labels_path = 'datasets/valences_landmark_complete.txt'
