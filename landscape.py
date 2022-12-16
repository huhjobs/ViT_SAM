import sys
import argparse
import random
from utils import str2bool
import os
import datetime
import pytz
import yaml
import copy
import pickle

import torch
import torch.nn as nn  
import timm

from load_data import load_data
from get_landscape import get_landscape, get_HDVW_landscape, get_Hessian_eig

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Options for loss landscape.')

    parser.add_argument('--HDVW', type=str2bool, required = True)
    parser.add_argument('--Hess', type=str2bool, required = False)
    parser.add_argument('--HDVW_n', type=int, default = 21)
    parser.add_argument('--HDVW_zlim', type=float, default = 0.5)
    parser.add_argument('--data_ratio', type=float, default = 0.1)
    parser.add_argument('--scale', type=float, default = 1e-0)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--path_to_model', type=str, required=True)
    parser.add_argument('--path_to_data', type=str, default='./Data/')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--steps', type=int, default=40)

    opt = parser.parse_args(sys.argv[1:])
    
## Setting ##
if torch.cuda.is_available():
    DEVICE = torch.device(f"cuda:{opt.gpu_num}")
else: DEVICE = torch.device('cpu')
    
## HYPERPARAMETER LOADING ##
path = f'./results/{opt.path_to_model[21:]}'

split = path.split('/')
model_name = split[2]
opt_name = split[3]
run_path = f'./results/{model_name}/{opt_name}/{split[4]}/'
config_path = f'./results/{model_name}/{opt_name}/{split[4]}/config.yaml'

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

for var in config.split('--'):
    if 'batch_size' in var:
        BATCH_SIZE = int(var[11:])

## LOSS ##
criterion = torch.nn.CrossEntropyLoss()

## LOAD DATA ##
data_loaders = load_data(opt.path_to_data, BATCH_SIZE, opt.image_size, ratio = opt.data_ratio)
train_loader, _ , _ = data_loaders

## LOAD MODELS ##
model = timm.create_model(model_name, pretrained=False)

if 'resnet' in model_name:
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,2)
    model.fc = nn.Sequential(model.fc, nn.Softmax(),)

elif 'vit' in model_name:
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs,2)
    model.head = nn.Sequential(model.head, nn.Softmax(),)

model = model.to(DEVICE)
model_initial = copy.deepcopy(model)

checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])
model_final = copy.deepcopy(model)

if opt.Hess:
    max_eigen_list = get_Hessian_eig(model, train_loader, criterion, run_path, DEVICE)
    
    with open(f"{run_path}/Hessian_eig_list", "wb") as fp:   #Pickling
        pickle.dump(max_eigen_list, fp)

else:
    ## DRAW_LANDSCAPE ##
    if opt.HDVW:
        get_HDVW_landscape(model, train_loader, run_path, data_ratio = opt.data_ratio, z_lim = opt.HDVW_zlim, scale = opt.scale, n = opt.HDVW_n) 

    elif opt.HDVW == False:
        get_landscape(model_initial, model_final, criterion, train_loader, run_path, STEPS = opt.steps)

## Save Loss Landscape to trained model folder##
print(f'Done saving. Check {run_path}')