
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:18 2017
@author: majin
"""
import numpy as np
import matplotlib.pyplot as plt

def readFile2Matrix(path):
    '''
    从文件中读取数据
    '''
    splitter = ","
    with open(path) as f:
        first_ele  =   True
        for data in f.readlines():
            data = data.strip('\n')     # 去掉每行的换行符，"\n"
            nums = data.split(splitter) # 按照splitter进行分割。
            #nums = nums[0:-1]           # 每行的最后数据不要
            nums = [float(x) for x in nums]# 将字符串转化为浮点型数据
            nums[-1] = -1                  #所有行的最后一列表示该点的类族
            ## 添加到 matrix 中。
            if first_ele:
                matrix = np.array(nums)
                first_ele = False
            else:
                matrix = np.c_[matrix,nums]
    return np.transpose(matrix)




def plotData(data,color,style):
    '''
    绘制数据，但是不显示，以便对对个数据在同一张图上绘制
    @datas 
        其中一个矩阵
    @color 
        这个矩阵在渲染时使用的颜色
    @style
        点的style
    '''
    colors =['b','y','g','o','b']
    if data.shape[1] == 3:
        plt.scatter(data[:,0], data[:,1],c=data[:,2],marker=style)
    else:
        plt.scatter(data[:,0], data[:,1],c=color,marker=style)
    
    
def show():
    '''
    显示绘制数据
    '''
    plt.show()