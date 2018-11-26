# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 11:35:02 2018

@author: Lzj
"""

import cv2
import numpy as np
#import matplotlib.pyplot as plt

Img = cv2.imread('lena.jpg')
#cv2.imshow('Lena',Img)
#cv2.waitKey(0)
row, col, dim = Img.shape
u = list()
sigma = list()
v = list()

for i in range(dim):
    u0, sigma0, v0 = np.linalg.svd(Img[:,:,i])
    u.append(u0)
    sigma.append(sigma0)
    v.append(v0)

u = np.array(u)
v = np.array(v)
sigma = np.array(sigma)


ImgNew = np.zeros(Img.shape)

def eigValPct(eigVals,percentage):
    sortArray=np.sort(eigVals) #使用numpy中的sort()对特征值按照从小到大排序
    sortArray=sortArray[::-1] #特征值从大到小排序
    arraySum=np.sum(sortArray) #数据全部的方差arraySum
    tempSum=0
    num=0
    for i in sortArray:
        tempSum+=i
        num+=1
        if tempSum>=arraySum*percentage:
            return num 

def psnr(target, ref, scale):
    # target:目标图像  ref:参考图像  scale:尺寸大小
    # assume RGB image
    target_data = np.array(target)
#    target_data = target_data[scale:-scale,scale:-scale]
 
    ref_data = np.array(ref)
#    ref_data = ref_data[scale:-scale,scale:-scale]
 
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = np.sqrt( np.mean(diff ** 2.) )
    return 20*np.log10(255.0/rmse)


def SVDpro(ImgNew,dim,N,u,v,sigma):
    for i in range(dim):
        Diag = np.diag(sigma[i,:N])
        tmp = np.dot(u[i,:,:N], Diag).dot(v[i,:N,:])
        ImgNew[:,:,i] = tmp
    ImgNew[ImgNew<0] = 0
    ImgNew[ImgNew>255] = 255
    Psnr = psnr(ImgNew[:,:,0],np.float32(Img[:,:,0]),8)
    ImgNew = np.uint8(ImgNew)
    print(Psnr)
    cv2.imshow('result',ImgNew)
    cv2.waitKey(0)
    cv2.imwrite('./construction/'+str(N)+'.png',ImgNew)

for j in range(10):
    N = int(col * (j+1) * 1.0/20)
    SVDpro(ImgNew,dim,N,u,v,sigma)
cv2.destroyAllWindows()


