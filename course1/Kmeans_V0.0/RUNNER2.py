# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:18 2017

根据中心点是否还在变化决定是否终止
@author: majin
"""
from Utils_v2 import * 
from KMeans_v2 import *

if __name__ == "__main__":
    filePath = "Aggregation.txt"
    K = 4   #集群数
    
    #读取文件
    nodes = readFile2Matrix(filePath)
    plotData(nodes,'b','.')
        
    #随机化K
    center = randK(K)
    plotData(center,'r','x')
    show()
    
    while 1:
        
        lastCenter = center
        
        #找最近的center进行归类
        nodes = cluster2near(nodes,center)
        #print(nodes)
        
        #更新center
        center = compute_center(nodes,K)
        #print(center)
        
        plotData(nodes,'b','.')
        plotData(center,'r','x')
        show()
        
        if np.array_equal(lastCenter,center):
            break
        
