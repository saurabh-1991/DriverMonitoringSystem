import os
from glob import glob
import random
import time
import tensorflow
import datetime
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 3 = INFO, WARNING, and ERROR
from tqdm import tqdm
import numpy as np
import pandas as pd
from IPython.display import FileLink
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from IPython.display import display, Image
import matplotlib.image as mpimg


from sklearn.datasets import load_files
#from keras.utils import np_utils
from keras import utils
from sklearn.utils import shuffle
from sklearn.metrics import log_loss

from tensorflow import keras

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D

from tensorflow.keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.vgg16 import VGG16

from data_analysis import *
from model import *
from data_augmentation import *

def main():
    print("******DMS V0.1******")
    dataset = load_data()
    img_rows = 64  # dimension of images
    img_cols = 64
    color_type = 1  # grey
    nb_test_samples = 200
    # loading train images
    x_train, x_test, y_train, y_test = read_and_normalize_train_data(img_rows, img_cols, color_type)

    # loading validation images
    test_files, test_targets = read_and_normalize_sampled_test_data(nb_test_samples, img_rows, img_cols, color_type)
    analyse_dataset(x_train, x_test)
    models_dir = "../data"
    # if not os.path.exists(models_dir):
    #     os.makedirs(models_dir)

    checkpointer = ModelCheckpoint(filepath=models_dir+'/dms_model_v0.1.keras',
                                   monitor='val_loss', mode='min',
                                   save_weights_only=True,
                                   verbose=1, save_best_only=True)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
    model = create_dms_model_v0_1(img_rows,img_cols,color_type)
    model.summary()
    # Compiling the model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # Training
    # Number of batch size and epochs
    batch_size = 40  # 40
    nb_epoch = 6  # 10
    history1 = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        callbacks=[checkpointer],
                        epochs=nb_epoch, batch_size=batch_size, verbose=1)

    # model.load_weights('saved_models/weights_best_vanilla.hdf5')
    print('History of the training', history1.history)

    #Evaluate Model
    score1 = model.evaluate(x_test, y_test, verbose=1)

    print('Loss: ', score1[0])
    print('Accuracy: ', score1[1] * 100, ' %')

    # With Data Augumentation

    train_datagen,test_datagen = augment_training_data()
    nb_train_samples = x_train.shape[0]
    nb_validation_samples = x_test.shape[0]
    training_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    validation_generator = test_datagen.flow(x_test, y_test, batch_size=batch_size)

    checkpointer2 = ModelCheckpoint(filepath=models_dir + '/dms_model_v0.1_with_Augmentation.keras',
                                   monitor='val_loss', mode='min',
                                   save_weights_only=True,
                                   verbose=1, save_best_only=True)
    history_v2 = model.fit_generator(training_generator,
                                     steps_per_epoch=nb_train_samples // batch_size,
                                     epochs=nb_epoch,
                                     verbose=1,
                                     validation_data=validation_generator,
                                     callbacks=[checkpointer2],
                                     validation_steps=nb_validation_samples // batch_size)

    # Evaluate and compare the performance of the new model
    score2 = model.evaluate_generator(validation_generator, nb_validation_samples // batch_size)
    print("Loss for model 1", score1[0])
    print("Loss for model 2 (data augmentation):", score2[0])

    print("Test accuracy for model 1", score1[1])
    print("Test accuracy for model 2 (data augmentation):", score2[1])
    model.save("../data/dms_model_v0.1_04_01_2024_aug.keras")


if __name__ == '__main__':
    main()

#https://www.kaggle.com/code/pierrelouisdanieau/computer-vision-tips-to-increase-accuracy