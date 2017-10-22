# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:18 2017

根据指定的迭代次数，决定是否停止
@author: majin
"""
from KMeans import *
from Utils import *

if __name__ == "__main__":
    filePath = "Aggregation.txt"
    K = 4   #集群数
    T = 50  #迭代次数
    
    #读取文件
    nodes = readFile2Matrix(filePath)
    plotData(nodes,'b','.')
        
    #随机化K
    center = randK(K)
    plotData(center,'r','x')
    show()
    
    for i in range(T):

        #找最近的center进行归类
        nodes = cluster2near(nodes,center)
        #print(nodes)
        
        #更新center
        center = compute_center(nodes,K)
        #print(center)
        
        plotData(nodes,'b','.')
        plotData(center,'r','x')
        show()