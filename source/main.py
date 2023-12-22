import os
import gc
import sys

import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

from skimage.feature import hog, canny
from skimage import color

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from tensorflow.python.keras import layers
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.models import Sequential, Model
# from tensorflow.keras.preprocessing import image
from keras_preprocessing import image
from tensorflow.python.keras.layers import Input, Dense, Activation, Dropout

from tensorflow.python.keras.layers import *
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.applications.imagenet_utils import preprocess_input
#from tensorflow.python.keras.applications.vgg19 import VGG19
#from tensorflow.python.keras.applications import ResNet50
from tf_explain.core.activations import ExtractActivations
from tf_explain.core.grad_cam import GradCAM
from sklearn.model_selection import train_test_split
from keras.utils.data_utils import get_file
from tensorflow.python.client import device_lib
#--------------------------------------------------------------------#
from data_prepare import *
from visualization import visualize_train_data,visualize_class_distribution_analysis
from image_data_analysis import visualize_hsv_images,visualize_edges_images_gray,visualize_corners_images_gray
from image_agumentations import plot_augimages,augment_training_data
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
    add_resolution_data(train_data)
    #print(train_data.sort_values('width').head(84))
    pred_df = load_gt_data(path_list)
    check_missing_data(train_data)
    #visualize_train_data(train_data)
    # visualize_class_distribution_analysis(train_data)
    # for count,class_name in enumerate(train_data['classname'].unique()):
    #     if count == 0:
    #         visualize_hsv_images(class_name, train_data)
    #         visualize_corners_images_gray(class_name,train_data)
    #     else:
    #         break
    datagen = augment_training_data()
    # plot_augimages(np.random.choice(train_data['path'],5),datagen)



if __name__ == "__main__":
    main()
