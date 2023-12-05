import pandas as pd
import numpy as np

path_list = ['/home/saurabh/Project/DMS/Dataset/Kaggle_Data/driver_imgs_list.csv',
             '/home/saurabh/Project/DMS/Dataset/Kaggle_Data/imgs/train/',
             '/home/saurabh/Project/DMS/Dataset/Kaggle_Data/sample_submission.csv']

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


def load_data(path):
    print(type(path))
    train_df = pd.read_csv(path[0])
    train_df['path'] = path[1] + train_df['classname'] + '/' + \
                       train_df['img']
    print(train_df.head(5))
    return train_df

def load_gt_data(path):
    pred_df = pd.read_csv(path[2])
    return pred_df
def data_analysis(ip_pd_frame):
    print(ip_pd_frame.count())
    print('Train samples count: ', len(ip_pd_frame))
    print(ip_pd_frame.columns)
    print('Class Count: ', len(ip_pd_frame['classname'].value_counts()))
    print(ip_pd_frame['classname'].value_counts())