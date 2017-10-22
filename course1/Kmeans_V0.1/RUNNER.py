# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:18 2017
使用Kmeans对点进行聚类
@author: majin
"""
import numpy as np
import matplotlib.pyplot as plt
from Utils import *
from KMeans import *


if __name__ == "__main__":
    filePath = "Aggregation.txt"
    
    #产生[2...maxNum],k值在其中进行遍历
    maxNum = 10
    Krange = [x for x in range(maxNum+1) if x > 1 ]
    
    all_res = np.zeros((1,maxNum+1)) #保存每个K的评价度，绘制折线
    #读取文件
    initNodes = readFile2Matrix(filePath) #(n,3)
    plotData(initNodes,'b','.')
    plt.title("the initial view of unclustering")
    plt.xlabel('x')
    plt.ylabel('y')
    show()
    
    for K in Krange:
        
        nodes = np.copy(initNodes) #获取node坐标
        center = randK(nodes,K) #随机化K
        
        T = 100  #迭代次数
        for i in range(T):
            
            #记录上一次的所有点
            lastNodes = np.copy(nodes)
            
            #找最近的center进行归类
            nodes = cluster2near(nodes,center)
            
            #更新center
            center = compute_center(nodes,K)
            
            #plotData(nodes,'b','.')
            #plotData(center,'r','x')
            #show()
            
            #点的类不变，视为结束
            if np.array_equal(lastNodes,nodes):
                break
         
        #计算当前K的聚集度
        eva = evaluate_result(center,nodes)             
        #print(eva)
        #保存记录
        all_res[0][K] = eva

        #展示当前K的聚类效果        
        plotData(nodes,'b','.')
        plotData(center,'r','x')
        
        plt.title('k='+str(K)+",the result of k-means clustering")
        plt.xlabel('x')
        plt.ylabel('y')
        
        show()


    #显示所有K的聚类的聚集度的折线图
    all_res = all_res[0][2:]
    plt.plot(Krange,all_res,'bo-')

    plt.title('all K with its clustering effect')
    plt.xlabel('K')
    plt.ylabel('distrabution')

    plt.show()