import numpy as np
import copy
import math
import random
import pylab as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os
import skimage
import numpy as np
from sklearn.metrics import accuracy_score,cohen_kappa_score,recall_score
from processing_library import *
from cv2 import *
from PIL import Image
import scipy.io as sio
import math
from processing_library_cnn import *
from metrics_torch import sam, psnr, scc, ergas,mse, SSIM
import random
from torch.autograd import Variable
from torchvision import models
from skimage.metrics import peak_signal_noise_ratio as PSNR
#DEB Dataset

dim_out_class = 5
learning_rate = 0.00005
batch_size = 16

class Residual_Block(nn.Module):
    def __init__(self, channels):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, stride=(1,1), out_channels=channels, kernel_size=(3,3), padding='same')
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=channels, stride=(1,1), out_channels=channels, kernel_size=(3,3), padding='same')

    def forward(self, x):
        fea = self.relu(self.conv1(x))
        fea = self.relu(self.conv2(fea))
        result = fea + x
        return result

def get_edge(data):
    data = data.cpu().numpy()
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5),
                                                        normalize=True)  # 第二个参数的-1表示输出图像使用的深度与输入图像相同
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5), normalize=True)
    return torch.from_numpy(rs)


class Multi_task_individual(nn.Module):
    def __init__(self):
        super(Multi_task_individual, self).__init__()
        self.f = None
        self.conv_redu = nn.Conv2d(dim_input + 1, 32, kernel_size=(3, 3), padding='same', stride=(1, 1))
        self.conv_back = nn.Conv2d(32, dim_input, kernel_size=(3, 3), padding='same', stride=(1, 1))
        # self.p1 = nn.parameter(torch.ones(3,3))
        self.block = nn.Sequential(
            Residual_Block(32),
            Residual_Block(32),
            # Residual_Block(32),
            # Residual_Block(32),
        )
        self.loss = torch.nn.L1Loss()
        ###################分类
        self.class_redu = nn.Conv2d(27, 32, kernel_size=(3, 3), padding='same', stride=(1, 1))
        self.conv = nn.Conv2d(32, 128, kernel_size=(3, 3), padding='same', stride=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding='same', stride=1)
        self.pool = nn.MaxPool2d((2, 2), stride=2, padding=1)
        self.fc = nn.Linear(6400, dim_out_class)
        self.cross_loss = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x_LRHS, x_PAN, x_class, gt_pan=None, gt_class=None):
        h_lrhs = get_edge(x_LRHS)
        h_pan = get_edge(x_PAN)
        re_h_lrhs = torch.nn.functional.interpolate(h_lrhs, size=[H, W], mode='bicubic')
        re_lrhs = torch.nn.functional.interpolate(x_LRHS, size=[H, W], mode='bicubic')
        mixed = torch.concat([re_h_lrhs, h_pan], dim=1).to(device)
        mixed1 = self.conv_redu(mixed)
        x = mixed1
        x = self.block(x)
        x = self.conv_back(x)
        last = x + re_lrhs
#######################################################
        f_class = self.class_redu(x_class)
        f_class = self.pool(f_class)
        f_class = self.block(f_class)
        f_class = self.conv(f_class)
        f_class = self.pool(f_class)
        f_class = self.conv2(f_class)
        fc = self.fc(f_class.reshape(-1, f_class.shape[1] * f_class.shape[2] * f_class.shape[3]))

        if gt_pan is None:
            return last, fc
        else:
            loss_pan = self.loss(last, gt_pan)
            loss_class = self.cross_loss(fc,gt_class.long()-1)
            loss = loss_pan + loss_class

        self.f = [loss_pan, loss_class]
        return loss_pan, loss_class, loss

def load_new_train():
    pre = np.load('F:\wxd_第四个工作\multitask_Pavia\LR_HSI_Class\Pavia_train.npz')
    train_LR = np.transpose(pre['train_LR'],[0, 3, 1, 2])
    train_PANI = np.expand_dims(pre['train_PANI'],axis= 1)
    train_gt_cla = np.squeeze(pre['train_gt_cla'])
    train_gt_pan = np.transpose(pre['train_gt_pan'],[0, 3, 1, 2])
    train_class = np.transpose(pre['train_class'], [0, 3, 1, 2])
    return train_LR,  train_PANI, train_gt_cla, train_gt_pan, train_class[:,:,:26,:26]

train_LR, train_PANI, train_gt_cla, train_gt_pan, train_class = load_new_train()
dim_input, H, W = train_LR.shape[1], train_gt_pan.shape[2], train_gt_pan.shape[3]
dim_out_class = int(np.max(train_gt_cla))

def BestValue(N):
    f = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        for name, param in net.named_parameters():
            temp = globals()['params' + str(i)][name]
            bb = torch.tensor(temp)
            net.state_dict()[name].copy_(bb)
        ####评估单个粒子的适应度
        index = batch_size
        while index < train_LR.shape[0]:
            batch_LRHS, batch_PAN, batch_y = next_batch_pansharp(train_LR, train_PANI, train_gt_pan, index, batch_size)
            batch_x_class = torch.tensor(next_batchx(train_class, index), dtype=torch.float32).to(device)
            batch_y_class = torch.from_numpy(next_batchy(train_gt_cla, index).astype(np.float32)).to(device)
            batch_LRHS = torch.tensor(batch_LRHS, dtype=torch.float32).to(device)
            batch_PAN = torch.tensor(batch_PAN, dtype=torch.float32).to(device)
            batch_y = torch.from_numpy(batch_y.astype(np.float32)).to(device)
            loss = net(batch_LRHS, batch_PAN, batch_x_class, batch_y, batch_y_class)
            index = index + batch_size
            f[i,0],f[i,1] = f[i,0] + loss[0], f[i,1]+ loss[1]
    best = np.min(f,axis=0)
    return best

def Initial(N):
    #initialize the population and the weight vector lambda list
    Lamb=[]
    for i in range(N):
        temp=[]
        temp.append(float(i)/(N))
        temp.append(1.0-float(i)/(N))
        Lamb.append(temp)  ###生成均匀的权重向量

    return Lamb

def Dominate(x,y,min=True):
    if min:

        for i in range(2):
            if x[i]>y[i]:
                return False

        return True
    else:
        for i in range(2):
            if x[i]<y[i]:
                return False
        return True
def Tchebycheff(x,lamb,z):
    #Tchebycheff approach operator

    temp=[]
    for i in range(2):
        temp.append(np.abs(x[i]-z[i])*lamb[i])
    return np.max(temp)
class My_loss(nn.Module):
    def __init__(self):
        super(My_loss, self).__init__()
        self.loss = torch.nn.L1Loss()
        self.cross_loss = nn.CrossEntropyLoss()

    def forward(self,x1,fc,y1,gt_class,lamb):
        return self.loss(x1,y1),self.cross_loss(fc,gt_class.long()-1), lamb[0]* self.loss(x1,y1) +lamb[1] * self.cross_loss(fc,gt_class.long()-1)


def train_net(lamb,train_LR,train_PANI,train_class,train_gt_pan,train_gt_cla):
    num_epoch = 0
    epoch = 3
    index = batch_size
    criterion = My_loss()
    while num_epoch < epoch:
        batch_LRHS, batch_PAN, batch_y = next_batch_pansharp(train_LR, train_PANI, train_gt_pan, index, batch_size)
        batch_x_class = torch.tensor(next_batchx(train_class, index), dtype=torch.float32).to(device)
        batch_y_class = torch.from_numpy(next_batchy(train_gt_cla, index).astype(np.float32)).to(device)
        batch_LRHS = torch.tensor(batch_LRHS, dtype=torch.float32).to(device)
        batch_PAN = torch.tensor(batch_PAN, dtype=torch.float32).to(device)
        batch_y = torch.from_numpy(batch_y.astype(np.float32)).to(device)
        net.optimizer.zero_grad()
        output1 = net(batch_LRHS, batch_PAN, batch_x_class)
        loss1 = criterion(output1[0], output1[1], batch_y, batch_y_class, lamb)
        loss1[2].backward()
        net.optimizer.step()  # 更新分类层权值
        index = index + batch_size
        if index > (train_LR.shape[0]):
            index = batch_size
            index_train = np.arange(train_LR.shape[0])
            np.random.shuffle(index_train)
            train_LR, train_PANI, train_gt_pan = train_LR[index_train, :, :, :], train_PANI[index_train, :,
                                                                                 :], train_gt_pan[index_train,
                                                                                     :, :, :]
            train_class, train_gt_cla = train_class[index_train, :, :, :], train_gt_cla[index_train]
            num_epoch = num_epoch + 1
    index = batch_size
    ####评估单个粒子的适应
    with torch.no_grad():
        f1 = np.zeros((1,2))
        while index < train_LR.shape[0]:
            batch_LRHS, batch_PAN, batch_y = next_batch_pansharp(train_LR, train_PANI, train_gt_pan, index, batch_size)
            batch_x_class = torch.tensor(next_batchx(train_class, index), dtype=torch.float32).to(device)
            batch_y_class = torch.from_numpy(next_batchy(train_gt_cla, index).astype(np.float32)).to(device)
            batch_LRHS = torch.tensor(batch_LRHS, dtype=torch.float32).to(device)
            batch_PAN = torch.tensor(batch_PAN, dtype=torch.float32).to(device)
            batch_y = torch.from_numpy(batch_y.astype(np.float32)).to(device)

            output1 = net(batch_LRHS, batch_PAN, batch_x_class)
            loss1 = criterion(output1[0], output1[1], batch_y, batch_y_class, lamb)
            index = index + batch_size
            f1[0,0], f1[0,1] =  f1[0,0] + loss1[0], f1[0,1] + loss1[1]
    return f1

def test_net(lamb,train_LR,train_PANI,train_class,train_gt_pan,train_gt_cla):
    ####评估单个粒子的适应
    index = batch_size
    criterion = My_loss()
    with torch.no_grad():
        f1 = np.zeros((1,2))
        while index < train_LR.shape[0]:
            batch_LRHS, batch_PAN, batch_y = next_batch_pansharp(train_LR, train_PANI, train_gt_pan, index, batch_size)
            batch_x_class = torch.tensor(next_batchx(train_class, index), dtype=torch.float32).to(device)
            batch_y_class = torch.from_numpy(next_batchy(train_gt_cla, index).astype(np.float32)).to(device)
            batch_LRHS = torch.tensor(batch_LRHS, dtype=torch.float32).to(device)
            batch_PAN = torch.tensor(batch_PAN, dtype=torch.float32).to(device)
            batch_y = torch.from_numpy(batch_y.astype(np.float32)).to(device)

            output1 = net(batch_LRHS, batch_PAN, batch_x_class)
            loss1 = criterion(output1[0], output1[1], batch_y, batch_y_class, lamb)
            index = index + batch_size
            f1[0,0], f1[0,1] =  f1[0,0] + loss1[0], f1[0,1] + loss1[1]
    return f1

def Generate_next_gen(index ,lamb,train_LR,train_PANI,train_class,train_gt_pan,train_gt_cla):
    f = np.zeros((N, 2), dtype=np.float32)
    for i in range(int(N)):
        # print('para:',i)
        if i == index:
            for name, param in net.named_parameters():
                temp = globals()['params' + str(i)][name]
                bb = torch.tensor(temp)
                net.state_dict()[name].copy_(bb)
            f[i] = train_net(lamb[i], train_LR, train_PANI, train_class, train_gt_pan, train_gt_cla)
            for name, param in net.named_parameters():
                globals()['params' + str(i)][name] = param.detach().cpu().numpy()
        else:
            for name, param in net.named_parameters():
                temp = globals()['params' + str(i)][name]
                bb = torch.tensor(temp)
                net.state_dict()[name].copy_(bb)
            f[i] = test_net(lamb[i], train_LR, train_PANI, train_class, train_gt_pan, train_gt_cla)
    return f


def MOEAD(N,T,train_LR,train_PANI,train_class,train_gt_pan,train_gt_cla):
    #the main algorithm
    #N:population numbers
    #T:the number of neighborhood of each weight vector
    Lamb=Initial(N) #p是个体，Lamb是Lamb1和Lamb2的权重向量
    z=BestValue(N=N)
    EP=[]
    t=0
    while(t<10):
        EP=[]
        t+=1
        print ('iteration:',t)
        ###训练10次产生孩子
        for i in range(N):
            f = Generate_next_gen(i, Lamb, train_LR, train_PANI, train_class, train_gt_pan, train_gt_cla)
            if i == 0:
                y0 = f[0,:]
                y1 = f[1,:]
                y2 = [100,100]

            if 0 < i < N-1:
                y0 = f[i - 1, :]
                y1 = f[i, :]
                y2 = f[i + 1, :]

            if i == N-1:
                y0 = f[N-2,:]
                y1 = f[N-1,:]
                y2 = [100, 100]
            #####更新Z

            if Dominate(y0,y1):
                y=y0
            else:
                y=y1
            if Dominate(y2,y):
                y=y2
            else:
                y=y

            for j in range(2):
                if y[j] < z[j]:
                    z[j] = y[j]
           #######################################
            ########更新邻域解
            if i ==0:
                Ta = Tchebycheff(y0, Lamb[0], z)
                Tb = Tchebycheff(y1, Lamb[1], z)
                # Td = Tchebycheff(y, Lamb[i], z)
                index = np.argmin(np.vstack((Ta,Tb)))
                if Ta<Tb: ##4是邻居数，把待对比的例子放到stack的最后
                    for name, param in net.named_parameters():
                        temp = globals()['params' + str(0)][name]
                        globals()['params' + str(i)][name] = temp
            if i ==N-1:
                Ta = Tchebycheff(y1, Lamb[i-1], z)
                Tb = Tchebycheff(y2, Lamb[i], z)
                # Td = Tchebycheff(y, Lamb[i], z)
                if Tb< Ta:
                    for name, param in net.named_parameters():
                        temp = globals()['params' + str(i)][name]
                        globals()['params' + str(i-1)][name] = temp
            if 0 < i <N-1:
                Ta = Tchebycheff(y0, Lamb[i-1], z)
                Tb = Tchebycheff(y1, Lamb[i], z)
                Tc = Tchebycheff(y2, Lamb[i+1], z)
                index = np.argmin(np.vstack((Ta,Tb,Tc)))
                if index == 2: ##4是邻居数，把待对比的例子放到stack的最后
                    for name, param in net.named_parameters():
                        temp = globals()['params' + str(i-1)][name]
                        globals()['params' + str(i)][name] = temp
                    for name, param in net.named_parameters():
                        temp = globals()['params' + str(index+i-1)][name]
                        globals()['params' + str(i)][name] = temp
            EP.append(y)

        x = []
        y = []
        for i in range(len(EP)):
            x.append(EP[i][0])
            y.append(EP[i][1])
        pl.plot(x, y, '*')
        pl.xlabel('f1')
        pl.ylabel('f2')
        pl.savefig('F:\wxd_第四个工作\multitask_Pavia/'+ str(t)+ 'code.jpg')
    return EP

N = 50 #个体数
B = 3 ##邻居的个数
s_window = 40
ratio = 3
Parameter = {}
net = Multi_task_individual()
for i in range(N):
    locals()['net'+str(i)] = Multi_task_individual()
    locals()['params'+str(i)] = {}
    for name, param in locals()['net'+str(i)].named_parameters():
        locals()['params'+str(i)][name] = param.detach().cpu().numpy()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Multi_task_individual()
net.to(device)
EP = MOEAD(N, B, train_LR, train_PANI, train_class, train_gt_pan, train_gt_cla)

for i in range(N):
    temp = globals()['params'+str(i)]
    np.save('params'+str(i), temp)

np.save('EP', EP)

