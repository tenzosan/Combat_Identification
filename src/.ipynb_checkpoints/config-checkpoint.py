# ==================================== Standard Libraries ==================================== #

import os
import warnings
import random as rn

# ==================================== Data Handling ==================================== #

import numpy as np
import pandas as pd
import joblib

# ==================================== Statistics ==================================== #

from scipy.stats import skew, kurtosis

# ==================================== Scikit-learn ==================================== #

from sklearn import svm, tree, metrics

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ==================================== TensorFlow / Keras ==================================== #

import tensorflow as tf

from tensorflow.keras.models import (
    Sequential,
    load_model
)

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LSTM
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# ==================================== Visualization ==================================== #

import matplotlib.pyplot as plt

# ==================================== Global Configuration ==================================== #

RANDOM_STATE = 1

# Random Seeds
rn.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Suppress Warnings
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning
)