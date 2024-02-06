# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 20:38:39 2018

@author: Jiantong Chen
"""

import os
import copy
import cv2
import scipy.io as sio
import numpy as np
import math
from sklearn import preprocessing
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.linear_model import OrthogonalMatchingPursuit
import scipy.stats as st

###############################################################################
def show_hsi(fuse,flag):
    if flag == 1:#pavia center
        fuse[fuse<0]=0
        fuse[fuse>1]=1
        rgb_datas = fuse[:, :, (80, 40, 20)]
        # rgb_datas = fuse[:, :, (25, 15, 20)]
        bgr_datas = rgb_datainputs[:, :, (2, 1, 0)]
        bgr_datas -= np.min(bgr_datas)
        bgr_datas /= np.max(bgr_datas)
        bgr_datas *= 255
        return bgr_datas
    if flag == 2:  # Botswana
        fuse[fuse<0]=0
        fuse[fuse>1]=1
        rgb_datas = fuse[:, :, (80, 90, 100)]
        bgr_datas = rgb_datas[:, :, (2, 1, 0)]
        bgr_datas -= np.min(bgr_datas)
        bgr_datas /= np.max(bgr_datas)
        bgr_datas *= 255
        return bgr_datas
    if flag == 3:  # Chikusei
        fuse[fuse<0]=0
        fuse[fuse>1]=1
        rgb_datas = fuse[:, :, (80, 40, 20)]
        bgr_datas = rgb_datas[:, :, (2, 1, 0)]
        bgr_datas -= np.min(bgr_datas)
        bgr_datas /= np.max(bgr_datas)
        bgr_datas *= 255
        return bgr_datas

def upsample_bicubic(image, ratio):
    h, w, c = image.shape
    re_image = cv2.resize(image, (w * ratio, h * ratio), cv2.INTER_CUBIC)

    return re_image

def get_oa(data,X_valid_loc,Y_valid):
    size = np.shape(X_valid_loc)
    num = size[0]
    index_all = 0
    step_ = 5000
    y_pred = []
    while index_all<num:
        if index_all + step_ > num:
            input_loc = X_valid_loc[index_all:,:]
        else:
            input_loc = X_valid_loc[index_all:(index_all+step_), :]
        input = windowFeature(data,input_loc,w)
        input = torch.tensor(input.transpose([0, 3, 1, 2]), dtype=torch.float32).to(device)
        temp1,_,_ = net(input)
        temp1 = temp1.cpu().numpy()
        y_pred1=contrary_one_hot(temp1).astype('int32')
        y_pred.extend(y_pred1)
        index_all += step_
    return y_pred

def next_batchx(image,batch_size):
    start = batch_size-16
    end = batch_size
    return image[start:end,:,:,:]

def next_batchy(lable,batch_size):
    start = batch_size-16
    end = batch_size
    return lable[start:end]

def contrast_loss(labels, logits, batch_size):
    labels = tf.argmax(labels, axis=1)
    labels = tf.reshape(labels, (-1, 1))
    # indicator for yi=yj
    mask = tf.cast(tf.equal(labels, tf.transpose(labels)), tf.float32)

    # (zi dot zj) / temperature
    anchor_dot_contrast = tf.math.divide(
        tf.linalg.matmul(logits, tf.transpose(logits)),
        tf.constant(1.0, dtype=tf.float32))

    # for numerical stability
    logits_max = tf.math.reduce_max(anchor_dot_contrast, axis=-1, keepdims=True)
    anchor_dot_contrast = anchor_dot_contrast - logits_max

    # tile mask for 2N images
    # mask = tf.tile(mask, (2, 2))

    # indicator for i \neq j
    logits_mask = tf.ones_like(mask) - tf.eye(batch_size)
    mask *= logits_mask

    # compute log_prob
    # log(\exp(z_i \cdot z_j / temperature) / (\sum^{2N}_{k=1} \exp{z_i \cdot z_k / temperature}))
    # = (z_i \cdot z_j / temperature) - log(\sum^{2N}_{k=1} \exp{z_i \cdot z_k / temperature})
    # apply indicator for i \neq k in denominator
    exp_logits = tf.math.exp(anchor_dot_contrast) * logits_mask
    log_prob = anchor_dot_contrast - tf.math.log(tf.math.reduce_sum(exp_logits, axis=-1, keepdims=True))
    mean_log_prob = tf.reduce_sum(mask * log_prob, axis=-1) / tf.reduce_sum(mask, axis=-1)
    loss = -tf.reduce_mean(tf.reshape(mean_log_prob, (2, 64)))
    return loss

###############################################################################
def get_blur(data):
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape)==3:
            rs[i,:,:] =  cv2.GaussianBlur(data[i,:,:],(5,5),0)
        else:
            rs[i,:,:,:] = cv2.GaussianBlur(data[i,:,:,:],(5,5),0)
    return rs
###############################################################################
def normalizeData(data):
    ''' 原始数据归一化处理（每个特征） '''
    data_norm = np.zeros(np.shape(data));
    for i in range(np.shape(data)[2]):
        x = preprocessing.normalize(data[:, :, i].reshape([1, -1]))
        data_norm[:, :, i] = x.reshape([data.shape[0], data.shape[1]])
    return data_norm
###############################################################################
def PanSharpen(data):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32)
    data_sharpen = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        data_sharpen[i,:,:] = cv2.filter2D(data[i,:,:],-1,kernel)
    return data_sharpen
###############################################################################
def load_data(data_name):
    '''读取数据'''
    path = os.getcwd()
    pre = sio.loadmat(path + '/data/' + data_name + '/' + data_name + '_pre.mat')
    
    data_norm = pre['data_norm']
    labels_ori = pre['labels_ori']
    x_train = pre['train_x']
    y_train = pre['train_y'][0]
    train_loc = pre['train_loc']
    x_test = pre['test_x']
    y_test = pre['test_y'][0]
    test_loc = pre['test_loc']
    
    return data_norm,labels_ori,x_train,y_train,train_loc,x_test,y_test,test_loc
###############################################################################
def windowFeature(data, loc, w ):
    '''从扩展矩阵中得到窗口特征'''
    # loc = np.transpose(loc,[1,0])
    size = np.shape(data)
    # print(size)
    data_expand = np.zeros((int(size[0]+w),int(size[1]+w),size[2]),dtype=np.float16)
    newdata = np.zeros((len(loc[0]), w, w,size[2]),dtype=np.float16)
    for j in range(size[2]):    
        data_expand[:,:,j] = np.lib.pad(data[:,:,j], ((int(w / 2), int(w / 2)), (int(w / 2),int(w / 2))), 'symmetric')
        newdata[:,:,:,j] = np.zeros((len(loc[0]), w, w))
        for i in range(len(loc[0])):
            loc1 = loc[0][i]
            loc2 = loc[1][i]
            f = data_expand[loc1:loc1 + w, loc2:loc2 + w,j]
            newdata[i, :, :,j] = f
    return np.transpose(newdata,[0,3,1,2])
###############################################################################
def one_hot(lable,class_number):
    '''转变标签形式'''
    one_hot_array = np.zeros([len(lable),class_number])
    for i in range(len(lable)):
        one_hot_array[i,int(lable[i]-1)] = 1
    return one_hot_array
###############################################################################
def disorder(X,Y):
    '''打乱顺序'''
    index_train = np.arange(X.shape[0])
    np.random.shuffle(index_train)
    X = X[index_train, :]
    Y = Y[index_train, :]
    return X,Y
###############################################################################
def next_batch(image,lable,batch_size):
    '''数据分批'''
    start = batch_size-128
    end = batch_size
    return image[start:end,:,:,:],lable[start:end]
###############################################################################
def conv_layer_same(x,W,B,stride):
    '''不改变特征图尺寸的卷积'''
    x = tf.nn.conv2d(x,W,stride,padding='SAME',name='CONV')
    h = tf.nn.bias_add(x,B)
    bn = tf.contrib.layers.batch_norm(h,decay=0.9,epsilon=1e-5,scale=True,is_training=True)
    convout = tf.nn.relu(bn)
    return convout
###############################################################################
def conv_layer_valid(x,W,B,stride):
    '''改变特征图尺寸的卷积'''
    x = tf.nn.conv2d(x,W,stride,padding='VALID',name='CONV')
    h = tf.nn.bias_add(x,B)
    bn = tf.contrib.layers.batch_norm(h,decay=0.9,epsilon=1e-5,scale=True,is_training=True)
    convout = tf.nn.relu(bn)
    return convout
###############################################################################
def contrary_one_hot(label):
    '''将onehot标签转化为真实标签'''
    size=len(label)
    label_ori=np.empty(size)
    for i in range(size):
        label_ori[i]=np.argmax(label[i])+1
    return label_ori
###############################################################################
def save_result(data_name,oa,aa,kappa,per_class_acc,train_time,test_time):
    '''将实验结果保存在txt文件中'''
    write_content='\n'+data_name+'\n'+'oa:'+str(oa)+' aa:'+str(aa)+' kappa:'+str(kappa)+'\n'+'per_class_acc:'+str(per_class_acc)+'\n'+'train_time:'+str(train_time)+' test_time:'+str(test_time)+'\n'
    f = open(os.getcwd()+'/实验结果.txt','a')
    f.writelines(write_content)
    f.close()
    return       