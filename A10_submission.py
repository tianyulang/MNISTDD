import numpy as np
import random

n_epochs = 80
batch_size = 250 
lr = 0.002 
weight_decay = 0.0001 
log_interval = 100
import os
import sys
import re
import time
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data.sampler import *
from torch.utils.data import Dataset
from torchvision import transforms, datasets
if  torch.cuda.is_available():
    device = torch.device("cuda")
    print('Training on GPU: {}'.format(torch.cuda.get_device_name(0)))
else:
    device = torch.device("cpu")
    print('Training on CPU')
class bboxNet(nn.Module):

    def __init__(self):
        ''' Define all needed layers '''
        super(bboxNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16384, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, 10)
        self.fc6 = nn.Linear(10, 4)

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.xavier_normal_(self.conv4.weight)
        nn.init.xavier_normal_(self.conv5.weight)
        nn.init.xavier_normal_(self.conv6.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.fc5.weight)
        nn.init.xavier_uniform_(self.fc6.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x)) 
        x = F.relu(self.conv2(x))
        x = self.bn1(x) 
        x = self.pool1(x) 

        x = F.relu(self.conv3(x)) 
        x = F.relu(self.conv4(x)) 
        x = self.bn2(x)
        x = self.pool2(x) 

        x = F.relu(self.conv5(x)) 
        x = F.relu(self.conv6(x)) 
        x = self.bn3(x)
        x = self.pool3(x) 

        x = x.view(x.size(0), -1) 

        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x)) 
        x = F.relu(self.fc3(x)) 
        x = F.relu(self.fc4(x)) 
        x = F.relu(self.fc5(x)) 
        x = F.relu(self.fc6(x)) 

        return x


class classNet(nn.Module):
    def __init__(self):
        super(classNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)


def get_empty_image(batchsize):
    return  torch.zeros(batchsize, 1, 28, 28).to(device)


def getImage(batch_size,data,each_box,empty_image):
    for i in range(batch_size):
        empty_image[i,0] = data[i, 0, each_box[i, 0]:each_box[i, 2], each_box[i, 1]:each_box[i, 3]]
    return empty_image


def train(train_data, targets, bboxes):
    train_losses_bbox = []
    train_counter_bbox = []
    train_losses_classification = []
    train_counter_classification = []
    bbox_model = bboxNet().to(device)
    classification_model = classNet().to(device)

    optimizer1 = torch.optim.Adam(bbox_model.parameters(), lr = lr, weight_decay=weight_decay)
    optimizer2 = torch.optim.Adam(classification_model.parameters(), lr = lr, weight_decay=weight_decay)

    bboxes = bboxes.view(train_data.shape[0], 8).to(torch.float32)
    points = torch.cat((bboxes[:, 0:2], bboxes[:, 4:6]), dim=1)
    train_data = train_data.view(train_data.shape[0], 1, 64, 64).to(torch.float32)
    targets = targets.long()
    

    train_set = torch.utils.data.TensorDataset(train_data,targets,points)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size =batch_size,shuffle = True )

    bbox_model.train()
    classification_model.train()
    min_loss_bbox = 500
    min_loss_classification = 20

    for epoch in range(n_epochs):
        for batch_idx, (data, target,target_points) in enumerate(train_loader):
            
            batch = data.size(0)
            optimizer1.zero_grad()
            output_points = bbox_model(data)
            loss = F.mse_loss(output_points, target_points.to(device), True)
            loss.backward()
            optimizer1.step()

            output_points= output_points.int()
            output_points = output_points - ((output_points < 0).int() *  output_points)
            output_points = ((output_points > 36).int() * 36 +(output_points < 36).int() * output_points )            
            
            output_points = output_points.view(output_points.size(0), 2, 2)
            output_bboxes = torch.cat((output_points, output_points+28), dim=2)
            output_bbox1 = output_bboxes[:, 0] 
            output_bbox2 = output_bboxes[:, 1] 
            output_bbox1_image = getImage(batch,data,output_bbox1,get_empty_image(batch))
            output_bbox2_image = getImage(batch,data,output_bbox2,get_empty_image(batch))
            
            optimizer2.zero_grad()
            output_number1 = classification_model(output_bbox1_image)
            class_loss1 = F.nll_loss(output_number1, target[:, 0])
            class_loss1.backward()
            optimizer2.step()
            
            optimizer2.zero_grad()
            output_number2 = classification_model(output_bbox2_image)
            class_loss2 = F.nll_loss(output_number2, target[:, 1])
            class_loss2.backward()
            optimizer2.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tbboxLoss: {:.6f}\tclassificaionLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),class_loss1.item() + class_loss2.item()))
                train_losses_bbox.append(loss.item())
                train_counter_bbox.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
                train_losses_classification.append(class_loss1.item() + class_loss2.item())
                train_counter_classification.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
                if min_loss_bbox >loss.item():
                    # torch.save(bbox_model, '/content/drive/My Drive/A10/bbox_model.pt')
                    torch.save(bbox_model.state_dict(), '/content/drive/My Drive/A10/bbox_model.pt')
                    print("Save Model bbox")
                    min_loss_bbox = loss.item()
                if min_loss_classification > class_loss1.item() + class_loss2.item():
                    torch.save(classification_model.state_dict(), '/content/drive/My Drive/A10/classification_model.pt')
                    print("Save Model classification")
                    min_loss_classification = class_loss1.item() + class_loss2.item()
              

        if epoch == 4 and loss.item() > 80:
            print("bad beginning, restart training.")
            bbox_model, classification_model = train(train_data, targets, bboxes)
            break

    return bbox_model, classification_model
        


def validation(bbox_model, classification_model, pred_bboxes, pred_class, N, val_data):
    bbox_model.eval()
    classification_model.eval()

    val_data = torch.tensor(val_data).view(N, 1, 64, 64).to(torch.float32)
    val_data = val_data.to(device)

    
    with torch.no_grad():
        for i in range(N):
            
            data = val_data[i].view(1, 1, 64, 64)
            batch = data.size(0)
            output_points = bbox_model(data)
            
            output_points= output_points.int()
            output_points = output_points - ((output_points < 0).int() *  output_points)
            output_points = ((output_points > 36).int() * 36 +(output_points < 36).int() * output_points )
            
            output_points = output_points.view(output_points.size(0), 2, 2) 
            output_bboxes = torch.cat((output_points, output_points+28), dim=2)
            output_bbox1 = output_bboxes[:, 0] 
            output_bbox2 = output_bboxes[:, 1] 
            output_bbox1_image = getImage(batch,data,output_bbox1,get_empty_image(batch))
            output_bbox2_image = getImage(batch,data,output_bbox2,get_empty_image(batch))
            

            output_number1 = classification_model(output_bbox1_image)
            output_number2 = classification_model(output_bbox2_image)

            output_bboxes = output_bboxes.detach().cpu().numpy()
            output_number1 = output_number1.data.max(1, keepdim=False)[1][0]
            output_number2= output_number2.data.max(1, keepdim=False)[1][0]
            pred_class[i] = (min(output_number1, output_number2), max(output_number1, output_number2))
            pred_bboxes[i] = output_bboxes

    return pred_bboxes, pred_class



def classify_and_detect(images):
    """

    :param np.ndarray images: N x 4096 array containing N 64x64 images flattened into vectors
    :return: np.ndarray, np.ndarray
    """
    #if  torch.cuda.is_available():
        #device = torch.device("cuda")
        #print('Training on GPU: {}'.format(torch.cuda.get_device_name(0)))
    #else:
        #device = torch.device("cpu")
        #print('Training on CPU')
    bbox_model = bboxNet().to(device)
    classification_model = classNet().to(device)
    N = images.shape[0]

    # pred_class: Your predicted labels for the 2 digits, shape [N, 2]
    pred_class = np.empty((N, 2), dtype=np.int32)
    # pred_bboxes: Your predicted bboxes for 2 digits, shape [N, 2, 4]
    pred_bboxes = np.empty((N, 2, 4), dtype=np.float64)

    # add your code here to fill in pred_class and pred_bboxes
    #train_data = torch.tensor(np.load( './train_X.npy')).to(device)
    #train_label = torch.tensor(np.load( './train_Y.npy')).to(device)
    #train_box = torch.tensor(np.load('./train_bboxes.npy')).to(device)
    #bbox_model,classification_model = train(train_data, train_label, train_box)
    # torch.save(bbox_model, '/content/drive/My Drive/A10/bbox_model.pt')
    # torch.save(classification_model, '/content/drive/My Drive/A10/classification_model.pt')
    bbox_model.load_state_dict(torch.load('./A10/bbox_model.pt',map_location=torch.device('cpu')))
    classification_model.load_state_dict(torch.load('./A10/classification_model.pt',map_location=torch.device('cpu')))
    # bbox_model = torch.load('/content/drive/My Drive/A10/bbox_model.pt')
    # classification_model = torch.load('/content/drive/My Drive/A10/classification_model.pt')
    pred_bboxes, pred_class = validation(bbox_model, classification_model, pred_bboxes, pred_class, N, images)

    return pred_class, pred_bboxes