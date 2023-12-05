import os
import gc
import sys

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import skimage
from skimage.feature import hog, canny
from skimage.filters import sobel
from skimage import color

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from tensorflow.python.keras import layers
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.models import Sequential, Model
# from tensorflow.keras.preprocessing import image
from keras_preprocessing import image
from tensorflow.python.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import *
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.applications.imagenet_utils import preprocess_input
#from tensorflow.python.keras.applications.vgg19 import VGG19
#from tensorflow.python.keras.applications import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications import ResNet50
from tf_explain.core.activations import ExtractActivations
from tf_explain.core.grad_cam import GradCAM
from sklearn.model_selection import train_test_split
from keras.utils.data_utils import get_file
from tensorflow.python.client import device_lib
#--------------------------------------------------------------------#
from data_prepare import *
#--------------------------------------------------------------------#

# %matplotlib inline

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    print("[GPU] Information -{local_device_protos}")
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
def main():
    print('DMS Application v0.0')
    get_available_gpus()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print("We got a GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("Sorry, no GPU for you...")
    train_data = load_data(path_list)


if __name__ == "__main__":
    main()
