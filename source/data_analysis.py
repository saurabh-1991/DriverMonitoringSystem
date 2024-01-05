import cv2
import os
from glob import glob
import pandas as pd
from tqdm import tqdm
from tensorflow.python.keras.utils import np_utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import json

NUMBER_CLASSES = 10

path = {"TrainingImages": "/home/saurabh/Project/DMS/Dataset/Kaggle_Data/imgs/train/",
        "Labels": "/home/saurabh/Project/DMS/Dataset/Kaggle_Data/driver_imgs_list.csv",
        "TestImages": "/home/saurabh/Project/DMS/Dataset/Kaggle_Data/imgs/test"}

activity_map = {'c0': 'Safe driving',
                'c1': 'Texting - right',
                'c2': 'Talking on the phone - right',
                'c3': 'Texting - left',
                'c4': 'Talking on the phone - left',
                'c5': 'Operating the radio',
                'c6': 'Drinking',
                'c7': 'Reaching behind',
                'c8': 'Hair and makeup',
                'c9': 'Talking to passenger'}
def load_data():
    dataset = pd.read_csv(path['Labels'])
    dataset.head(5)
    by_drivers = dataset.groupby('subject')
    # Groupby unique drivers
    unique_drivers = by_drivers.groups.keys()  # drivers id
    print('There are : ', len(unique_drivers), ' unique drivers')
    print('There is a mean of ', round(dataset.groupby('subject').count()['classname'].mean()), ' images by driver.')
    return dataset


def get_cv2_image(path, img_rows, img_cols, color_type=3):
    """
    Function that return an opencv image from the path and the right number of dimension
    """
    if color_type == 1: # Loading as Grayscale image
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    elif color_type == 3: # Loading as color image
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_rows, img_cols)) # Reduce size
    return img


# Loading Training dataset
def load_train(img_rows, img_cols, color_type=3):
    """
    Return train images and train labels from the original path
    """
    train_images = []
    train_labels = []
    # Loop over the training folder
    for classed in tqdm(range(NUMBER_CLASSES)):
        print('Loading directory c{}'.format(classed))
        files = glob(os.path.join(path['TrainingImages']+'c' + str(classed), '*.jpg'))
        for file in files:
            img = get_cv2_image(file, img_rows, img_cols, color_type)
            #print("Image Shape:-",img.shape)
            train_images.append(img)
            train_labels.append(classed)
    return train_images, train_labels


def read_and_normalize_train_data(img_rows, img_cols, color_type):
    """
    Load + categorical + split
    """
    X, labels = load_train(img_rows, img_cols, color_type)
    y = np_utils.to_categorical(labels, 10)  # categorical train label
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)  # split into train and test
    x_train = np.array(x_train, dtype=np.uint8).reshape(-1, img_rows, img_cols, color_type)
    x_test = np.array(x_test, dtype=np.uint8).reshape(-1, img_rows, img_cols, color_type)

    return x_train, x_test, y_train, y_test


# Loading validation dataset
def load_test(size=200000, img_rows=64, img_cols=64, color_type=3):
    """
    Same as above but for validation dataset
    """
    test_path = os.path.join(path['TestImages'], '*.jpg')
    files = sorted(glob(test_path))
    X_test, X_test_id = [], []
    total = 0
    files_size = len(files)
    for file in tqdm(files):
        if total >= size or total >= files_size:
            break
        file_base = os.path.basename(file)
        img = get_cv2_image(file, img_rows, img_cols, color_type)
        X_test.append(img)
        X_test_id.append(file_base)
        total += 1
    return X_test, X_test_id


def read_and_normalize_sampled_test_data(size, img_rows, img_cols, color_type=3):
    test_data, test_ids = load_test(size, img_rows, img_cols, color_type)
    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.reshape(-1,img_rows,img_cols,color_type)
    return test_data, test_ids


def analyse_dataset(traindata,testdata):
    names = [item[17:19] for item in sorted(glob(path['TrainingImages']+"/*/"))]
    test_files_size = len(
        np.array(glob(os.path.join(path['TestImages'], '*.jpg'))))
    x_train_size = len(traindata)
    categories_size = len(names)
    x_test_size = len(testdata)
    print('There are %s total images.\n' % (test_files_size + x_train_size + x_test_size))
    print('There are %d training images.' % x_train_size)
    print('There are %d total training categories.' % categories_size)
    print('There are %d validation images.' % x_test_size)
    print('There are %d test images.' % test_files_size)

def write_json(data, fname='./output.json'):
    """Write data to json

    @param data: object to be written

    Keyword arguments:
    fname  -- output filename (default './output.json')

    """
    with open(fname, 'w') as fp:
        json.dump(data, fp)

def plot_train_history(history):
    """
    Plot the validation accuracy and validation loss over epochs
    """
    # Summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()