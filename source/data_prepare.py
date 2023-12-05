import pandas as pd
import numpy as np

path_list = ['/home/saurabh/Project/DMS/Dataset/Kaggle_Data/driver_imgs_list.csv'
             '/home/saurabh/Project/DMS/Dataset/Kaggle_Data/imgs/train/']

classes = {'c0': 'normal driving',
           'c1': 'texting - right',
           'c2': 'talking on the phone - right',
           'c3': 'texting - left',
           'c4': 'talking on the phone - left',
           'c5': 'operating the radio',
           'c6': 'drinking',
           'c7': 'reaching behind',
           'c8': 'hair and makeup',
           'c9': 'talking to passenger', }


def load_data(path_list):
    print(path_list.type())
    train_df = pd.read_csv(path_list[0])
    train_df['path'] = path_list[1] + train_df['classname'] + '/' + \
                       train_df['img']
