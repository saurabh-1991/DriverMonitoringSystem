import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import color
from skimage.filters import sobel
from data_prepare import classes


def visualize_hsv_images(class_name, train_data):
    classes_df = train_data[train_data['classname'] == class_name].reset_index(drop=True)
    for idx, i in enumerate(np.random.choice(train_data['path'], 2)):
        image = cv2.imread(i)
        hsv = color.rgb2hsv(image)
        dimension = hsv.shape
        fig = plt.figure(figsize=(8, 8))
        plt.suptitle(classes[class_name])
        plt.subplot(2, 2, 1)
        plt.imshow(image)
        plt.subplot(2, 2, 2)
        plt.imshow(hsv[:dimension[0], :dimension[1], 0], cmap="PuBuGn")
        plt.subplot(2, 2, 3)
        plt.imshow(hsv[:dimension[0], :dimension[1], 1], cmap='PuBuGn')
        plt.subplot(2, 2, 4)
        plt.imshow(hsv[:dimension[0], :dimension[1], 2], cmap='PuBuGn')
        plt.show()


def visualize_edges_images_gray(class_name,train_data):
    classes_df = train_data[train_data['classname'] ==  class_name].reset_index(drop = True)
    for idx,i in enumerate(np.random.choice(classes_df['path'],2)):
        image = cv2.imread(i)
        gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = sobel(image)
        gray_edges=sobel(gray)
        dimension = edges.shape
        fig = plt.figure(figsize=(8, 8))
        plt.suptitle(classes[class_name])
        plt.subplot(2,2,1)
        plt.imshow(gray_edges)
        plt.subplot(2,2,2)
        plt.imshow(edges[:dimension[0],:dimension[1],0], cmap="gray")
        plt.subplot(2,2,3)
        plt.imshow(edges[:dimension[0],:dimension[1],1], cmap='gray')
        plt.subplot(2,2,4)
        plt.imshow(edges[:dimension[0],:dimension[1],2], cmap='gray')
        plt.show()

def visualize_corners_images_gray(class_name,train_data):
    classes_df = train_data[train_data['classname'] ==  class_name].reset_index(drop = True)
    for idx,i in enumerate(np.random.choice(classes_df['path'],4)):
        image = cv2.imread(i)
        gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners_gray = cv2.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=0.02, minDistance=20)
        corners_gray = np.float32(corners_gray)
        for item in corners_gray:
            x, y = item[0]
            cv2.circle(image, (int(x), int(y)), 6, (0, 255, 0), -1)
        fig = plt.figure(figsize=(16, 16))
        plt.suptitle(classes[class_name])
        plt.subplot(2,2,1)
        plt.imshow(image, cmap="BuGn")
        plt.show()