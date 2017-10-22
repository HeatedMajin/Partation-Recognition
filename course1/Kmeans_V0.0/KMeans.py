# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:18 2017
@author: majin
"""
import numpy as np 
def randK(k):
    '''
    随机生成K个点
    '''
    rk = np.random.rand(k,2)
    al = np.array([30,40])
    al = al.reshape(2,1).T
    rk = np.abs(np.tanh(rk))*al
    return rk

def cluster2near(nodes,centers):
    '''
    将每个点指派到最近的质心，形成K个簇
    只改变node最后一列的的簇
    
    @nodes
        (n,3) n个点，第一列x坐标，第二列y坐标，第三列所属的类族
    @centers
        (k,2) k个中心点
    @res
        (n,3) nodes计算改变类族之后的结果
    '''
    res = nodes
    k = centers.shape[0]
    for i in range(nodes.shape[0]):      #per line(per node)
        node = np.array(nodes[i][0:-1])       
        node = node.reshape(1,2)         #(1,2)
        
        #计算到k个中心的距离
        distance = np.sum((node - centers)**2,axis=1,keepdims=True)#(k,1)
        #print(distance)
        
        #选出最小的距离，该中心就是该点的near
        minIndex = 0
        for index in range(k):
            if distance[index][0] < distance[minIndex][0]:
                minIndex = index
        
        #将这个点的类族归到距其最小的center上
        res[i][-1] = minIndex
        
    return res
        
def compute_center(nodes,k):
    '''
    根据节点的类族，计算每个族的中心
    @nodes
        (n,3) 所有节点
    @return 
        (k,2) 返回类族的中心
    '''
    n = nodes.shape[0]
    centers = np.zeros((k,2))
    count = np.zeros((k,1))
    for i in range(n):#per line(per node)
        node_class = int(nodes[i][-1])
        count[node_class][0] +=1
        centers[node_class][0] += nodes[i][0]
        centers[node_class][1] += nodes[i][1]
    centers = centers / count
    return centers