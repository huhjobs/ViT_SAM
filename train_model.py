import os
import numpy as np
import random

import torch
import torch.nn as nn  

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def train(DEVICE, sam, model, train_loader, BATCH_SIZE, Epoch, optimizer, criterion, log_interval):
    model.train()                                                 # assign train mode to the model
    
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    
    if sam == False:
      for batch_idx, (image, label) in enumerate(train_loader):
          image = image.to(DEVICE)
          label = label.to(DEVICE)
          optimizer.zero_grad()                                     # 과거에 이용한 Mini-Batch내에 있는 이미지 데이터와 레이블 데이터를 바탕으로 계산된 Loss의 Gradient값이 optimizer에 할당되어 있으므로 optimizer의 Gradient 초기화
          output = model(image)

          with torch.set_grad_enabled(True):
            loss = criterion(output, label)
            loss.backward()                                           # Back propagation으로 계산된 Gradient 값을 각 parameter에 할당
            optimizer.step()                                          # parameter update
            prediction = output.max(1, keepdim = True)[1]             # predicted labels in tensor
            
            train_acc.update(prediction.eq(label.view_as(prediction)).sum().item()/BATCH_SIZE)
            train_loss.update(loss.item())
            
            if batch_idx % log_interval == 0:                         # print log
                print("Train Epoch: {} [{}/{}({:.0f}%)]\tTrain Loss: {:.6f}".format(Epoch, batch_idx * len(image), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
    
    elif sam == True:
      for batch_idx, (image, label) in enumerate(train_loader):
          image = image.to(DEVICE)
          label = label.to(DEVICE)

          def closure():
              optimizer.zero_grad()
              output = model(image)
              loss = criterion(output, label)
              loss.backward()
              return loss

          loss = optimizer.step(closure)                      # parameter update
          prediction = model(image).max(1, keepdim = True)[1]             # predicted labels in tensor
            
          train_acc.update(prediction.eq(label.view_as(prediction)).sum().item()/BATCH_SIZE)
          train_loss.update(loss.item())
          
          if batch_idx % log_interval == 0:                         # print log
              print("Train Epoch: {} [{}/{}({:.0f}%)]\tTrain Loss: {:.6f}".format(Epoch, batch_idx * len(image), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))  
    
    return train_loss.avg, train_acc.avg




def evaluate(DEVICE, model, val_loader, BATCH_SIZE, criterion):
    model.eval()
    
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    
    with torch.no_grad():                                                     # 모델을 평가하는 단계에서는 Gradient를 통해 parameter값이 update되는 현상을 방지하기 위해 'torch.no_grad()' 메서드를 이용해 Gradient의 흐름을 억제
        for image, label in val_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            loss = criterion(output, label)
            prediction = output.max(1, keepdim = True)[1]                     # output값 (prediction probability)가 가장 높은 index(class)로 예측
    
            val_acc.update(prediction.eq(label.view_as(prediction)).sum().item()/BATCH_SIZE)
            val_loss.update(loss.item())
    
    return val_loss.avg, val_acc.avg  




def startTrain(DEVICE, model, data_loaders, BATCH_SIZE, optimizer, criterion, opt, EPOCHS,  TRAIN_PATIENCE):
    best_val_acc = 0
    best_val_loss = 100
    epoch = 0
    total_loss_train, total_acc_train = [],[]
    total_loss_val, total_acc_val = [],[]

    train_loader, val_loader, test_loader = data_loaders

    for Epoch in range(1, EPOCHS+1):
        train_loss, train_acc = train(DEVICE, opt, model, train_loader, BATCH_SIZE, Epoch, optimizer, criterion, log_interval = 200)
        total_acc_train.append(train_acc)
        total_loss_train.append(train_loss)

        val_loss, val_acc = evaluate(DEVICE, model, val_loader, BATCH_SIZE, criterion)
        total_acc_val.append(val_acc)
        total_loss_val.append(val_loss)
        print(Epoch)
        print("\n[EPOCH: {}], \tVal Loss: {:.4f}, \tVal Accuracy: {:.2f} %\n".format(Epoch, val_loss, val_acc))
        
        # monitoring test accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epoch = Epoch
            print('*****************************************************')
            print('best record: [epoch %d], [val loss %.5f], [val acc %.5f]' % (Epoch, val_loss, val_acc))
            print('*****************************************************')
        elif Epoch > epoch + TRAIN_PATIENCE:
            break
        
        # # monitoring val loss
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     epoch = Epoch
        #     print('*****************************************************')
        #     print('best record: [epoch %d], [val loss %.5f], [val acc %.5f]' % (Epoch, val_loss, val_acc))
        #     print('*****************************************************')
        # elif Epoch > epoch + TRAIN_PATIENCE:
        #     break

    return epoch, total_loss_train, total_acc_train, total_loss_val, total_acc_val, best_val_acc