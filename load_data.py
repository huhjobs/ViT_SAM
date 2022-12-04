import os
import numpy as np
import random

import cv2
import matplotlib.pylab as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

import seaborn as sns

import torch
import torch.nn as nn  
from torch.utils import data

import torchvision
from torchvision import transforms
random.seed(9000)


class Dataset(data.Dataset):
    
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __getitem__(self, index):
        image = self.images[index]
        label = int(self.labels[index])
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label
    
    def __len__(self):
        return len(self.images)
    

def load_data(data_dir, batchsize, imagesize):

    print('Data directory: ', data_dir)
    print('Batch size: ', batchsize)
    print('Image size: ', imagesize)

    resized_normal_arr = np.load(f'{data_dir}/{imagesize}_normal_arr.npy')
    resized_opacity_arr = np.load(f'{data_dir}/{imagesize}_opacity_arr.npy')

    # X_data and y_data
    X_data = np.concatenate((resized_normal_arr, resized_opacity_arr))
    y_data = []

    for i in range(len(resized_normal_arr)):
        y_data.append(0)   
    for i in range(len(resized_opacity_arr)):
        y_data.append(1)

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    # Balanced 8:1:1 Split
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state = 71, stratify = y_data)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/9, random_state = 71, stratify = y_train)

    train_dataset = Dataset(X_train, y_train, transform = transforms.ToTensor())
    val_dataset = Dataset(X_val, y_val, transform = transforms.ToTensor())
    test_dataset = Dataset(X_test, y_test, transform = transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                              batch_size = batchsize,
                                              shuffle = True)   
    val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                              batch_size = batchsize,
                                              shuffle = False)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                              batch_size = batchsize,
                                              shuffle = False)
    
    return train_loader, val_loader, test_loader
