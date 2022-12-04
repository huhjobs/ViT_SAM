import os
import numpy as np
import random

import torch
import torch.nn as nn  
from torch.utils import data

import torchvision
from torchvision import transforms

from sklearn.metrics import roc_curve, auc

from sampytorch.sam import SAMSGD 
import timm

import matplotlib.pyplot as plt

from pprint import pprint

# from models.network import Network

import copy
from tqdm import tqdm
from sklearn import metrics
from glob import glob

from load_data import load_data
from train_model import AverageMeter, train, evaluate, startTrain

import sys


def run(opt):
    use_gpu = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count()
    print("Available GPU count:" + str(gpu_count))
    
    NUM_EPOCHS = opt.n_epochs
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    ## Run Path
    if opt.eval_only:
        # test only. it is okay to have duplicate run_path
        os.makedirs(opt.run_path, exist_ok=True)
    else:
        # train from scratch, should not have the same run_path. Otherwise it will overwrite previous runs.
        try:
            os.makedirs(opt.run_path)
        except FileExistsError:
            print("[ERROR] run_path {} exists. try to assign a unique run_path".format(opt.run_path))
            return None, None
        except Exception as e:
            print("exception while creating run_path {}".format(opt.run_path))
            print(str(e))
            return None, None
    
    with open(os.path.join(opt.run_path,'config.yaml'),'w') as fp:
        fp.write('\n'.join(sys.argv[1:]))
    
    ## Load Data
    data_loaders = load_data(opt.path_to_data, opt.batch_size, opt.input_size)
    train_loader, val_loader, test_loader = data_loaders

    ## Load Architecture and Weights
    if opt.pretrained_path == 'False':
        model_ft = timm.create_model(opt.arch, pretrained=False)
    elif opt.pretrained_path == 'imagenet':
        model_ft = timm.create_model(opt.arch, pretrained=True)
    else:
        model_ft = timm.create_model(opt.arch, pretrained=False)
        model_ft.load_state_dict(torch.load(opt.pretrained_path))
        
    if "resnet" in opt.arch:
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,opt.num_classes)
        model_ft.fc = nn.Sequential(model_ft.fc, nn.Softmax(),)

    elif "vit" in opt.arch:
        num_ftrs = model_ft.head.in_features
        model_ft.head = nn.Linear(num_ftrs,opt.num_classes)
        model_ft.head = nn.Sequential(model_ft.head, nn.Softmax(),)
    else:
        f"recheck opt.arch = {opt.arch}"

    model = model_ft.to(device)

    ## Load Loss Function
    criterion = nn.CrossEntropyLoss()

    ## Load Optim
    if opt.optimizer == 'SGD':
        if opt.optimizer_plus == 'base':
            optt = False
            optimizer = torch.optim.SGD(model.parameters(), lr = opt.lr, momentum=opt.momentum)
        elif opt.optimizer_plus == 'sam':
            optt = True
            optimizer = SAMSGD(model.parameters(),lr = opt.lr, rho = opt.momentum)
    else:
        f"recheck opt.optimizer = {opt.optimizer}"

    ## Train
    epoch, total_loss_train, total_acc_train, total_loss_val, total_acc_val, best_val_acc = startTrain(device, model, data_loaders, opt.batch_size, optimizer, criterion, optt, opt.n_epochs, opt.patience)
    
    ## Save Model
    PATH = "{run_path}/model-epoch{epoch:02d}-loss_val{loss_val:.2f}-acc_val{acc_val:.2f}.pt".format(run_path = opt.run_path, epoch = epoch, loss_val = total_loss_val[epoch-1] , acc_val = total_acc_val[epoch-1])
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(), # trained parameters
            'optimizer_state_dict': optimizer.state_dict(), # this contains buffers and parameters that are updated as the model trains
            'accuracy':best_val_acc, 
            # 'loss': best_val_loss,
            'total_loss_train':total_loss_train,
            'total_acc_train':total_acc_train,
            'total_loss_val':total_loss_val,
            'total_acc_val':total_acc_val,
            }, PATH)
    
    print(f'Saved {opt.arch} model with {opt.optimizer_plus} optimizer to {PATH}')

    ## Evaluation
    true_labels = []
    predicted_labels = []
    predicted_values = []

    model.eval()
        
    test_loss = AverageMeter()
    test_acc = AverageMeter()

    with torch.no_grad():                                                     # 모델을 평가하는 단계에서는 Gradient를 통해 parameter값이 update되는 현상을 방지하기 위해 'torch.no_grad()' 메서드를 이용해 Gradient의 흐름을 억제
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            loss = criterion(output, label)
            prediction = output.max(1, keepdim = True)[1]                     # output값 (prediction probability)가 가장 높은 index(class)로 예측
            
            true_labels.append(label)
            predicted_labels.append(prediction)
            predicted_values.append(output)
            
            test_acc.update(prediction.eq(label.view_as(prediction)).sum().item()/opt.batch_size)
            test_loss.update(loss.item())

    # ROC Curve / AUC
    # tensor to array form
    true_test_lst = []
    predicted_test_lst = []
    predicted_value_lst = []

    for i in range(len(test_loader)):
        t_labels = true_labels[i].cpu().numpy()
        true_test_lst += list(t_labels)
        
        p_labels = predicted_labels[i].cpu().numpy()[:,0]
        predicted_test_lst += list(p_labels)
        
        p_values = predicted_values[i].cpu().numpy()
        predicted_value_lst += list(p_values)
        
    true_test_arr = np.array(true_test_lst)
    predicted_test_arr = np.array(predicted_test_lst)
    predicted_value_arr = np.array([list(arr) for arr in predicted_value_lst])
        
    true_test_arr[:20], predicted_test_arr[:20], predicted_value_arr[:20]

    # roc curve for classes

    fpr = {}
    tpr = {}
    thresh ={}

    for i in range(opt.num_classes):    
        fpr[i], tpr[i], thresh[i] = roc_curve(true_test_arr == i, predicted_value_arr[:, i])
        
    # # plotting
    # plt.plot(fpr[1], tpr[1], linestyle='--',color='blue', label='Positive: Opacity / Negative: Normal') # Sensitivity: 비정상 중에 비정상으로 predict된 비율 / Specificity: 정상 중에 정상으로 predict된 비율
    # plt.title('ROC curve')
    # plt.xlabel('1-Specificity')
    # plt.ylabel('Sensitivity')
    # plt.legend(loc='best')
    # plt.savefig(os.path.join(opt.run_path,'auc.png'))

    print("AUC: ", auc(fpr[1], tpr[1]))

    ## TODO : save eval results
    ## TODO : eval_only version
