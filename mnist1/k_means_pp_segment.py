# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 16:05:02 2017

@author: majin
"""


import numpy as np 
import random
import math 
import process_picture as pp
from PIL import Image
import matplotlib.pyplot as plt


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
    if data.shape[1] == 3:
        plt.scatter(data[:,0], data[:,1],c=data[:,2],marker=style)
    else:
        plt.scatter(data[:,0], data[:,1],c=color,marker=style)
    

def randK(nodes,k):
    '''
    随机选取一个点作为一个中心点，每次选取距离所有中心点最远的那个点作为中心点
    '''
    n = nodes.shape[0]
    f = random.randint(0,n)
    
    #选取第一个点
    centers = np.zeros((k,2))   #(k,2)
    centers[0][0] = nodes[f][0] #x坐标
    centers[0][1] = nodes[f][1] #y坐标
    centerIndex = []#保存在centers中的node序号，防止node重复添加
    centerIndex.append(f)
    
    d=1     #现在拥有的中心点的个数
    while d<k:
        #计算每个点到所有中心点的距离和
        dist = np.zeros((n,1))  #dist (n,1)
        
        for i in range(nodes.shape[0]):      #per line(per node)
            node = np.array(nodes[i][0:-1])       
            node = node.reshape(1,2)         #(1,2)
        
            #计算到d个中心的距离
            for ii in range(d):
                 tmp = np.sum((node - centers[ii].reshape(1,2))**2,axis=1,keepdims=True)#(1,1)
                 dist[i][0] = dist[i][0] + np.sqrt(tmp)#np.sum(np.sqrt(tmp),axis=0,keepdims=True)
            
        #选出最大的距离，该中心就是该点的near
        maxIndex = 0
        for index in range(n):
            if dist[index][0] > dist[maxIndex][0] and index not in centerIndex:
                maxIndex = index
                centerIndex.append(maxIndex)
        #将这个点加入到centers中
        centers[d][0] = nodes[maxIndex][0]
        centers[d][1] = nodes[maxIndex][1]
        
        d = d + 1
    return centers

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
    count = np.zeros((k,2))
    for i in range(n):#per line(per node)
        node_class = int(nodes[i][-1])
        count[node_class][0] +=1
        count[node_class][1] +=1
        centers[node_class][0] += nodes[i][0]
        centers[node_class][1] += nodes[i][1]
    #centers = centers[:][0] / count
    #centers = centers[:][1] / count
    centers = centers/count
    return centers


def evaluate_result(centers,nodes):
    '''
    评价聚类的结果
    @centers
        类的中心点 (k,2)
    @nodes
        被分类后的点 (n,3)
    @return
        结果为每类的紧凑度的和
        类的紧凑度：该类下所有点到中心点的距离的平均值
    '''
    n= nodes.shape[0]
    k = centers.shape[0]
    cluster_jincou = np.zeros((k,1))#(k,1) 每个类的紧凑
    cluster_count = np.zeros((k,1))#(k,1) 每个类下的点的个数
    
    for i in range(n):#per node
        node_cluster = nodes[i][2] #点所属的类
        node_cluster = int(node_cluster)
        #点到中心点的距离
        x_2_y_2 = (nodes[i][0]-centers[node_cluster][0])**2 + (nodes[i][1]-centers[node_cluster][1])**2
        node_dist = math.sqrt(x_2_y_2)
        
        cluster_jincou[node_cluster][0] += node_dist #记录距离和
        cluster_count[node_cluster][0] +=1 #记录点的个数
    cluster_jincou = cluster_jincou/cluster_count #距离平均
    #cluster_jincou= cluster_jincou
    res = np.sum(cluster_jincou,axis=0,keepdims=True) #求和
    
    
    #################感觉上面的有点问题，再加上中心点之间的距离
    #centers
    return res/k+0.5*k

def load_2img_nodes(img):
    '''
    将二值图像中非零点的坐标取出来
    
    img: Image
    '''
    width = img.width
    height = img.height
    
    img_data = img.getdata()
    img_data_matric = np.matrix(img_data)
    
    img_matric = img_data_matric.reshape(height,width)
    nodes =[]
    for x in range(img_matric.shape[1]):
        for y in range(img_matric.shape[0]):
            if img_matric[y,x] != 255:
                nodes.append([x,height - y,-1])#x,y,类别
    return np.array(nodes)
    
if __name__ =="__main__":
    filein = r'./image/tests.png'
    img = pp.openImage(filein)
    img_gray = pp.rgb2gray(img)
    nodes = load_2img_nodes(img_gray)
    
    K=8
    center = randK(nodes,K) #随机化K
    
    T = 1000  #迭代次数
    for i in range(T):
        
        #记录上一次的所有点
        lastNodes = np.copy(nodes)
        
        #找最近的center进行归类
        nodes = cluster2near(nodes,center)
        
        #更新center
        center = compute_center(nodes,K)
        
        #点的类不变，视为结束
        if np.array_equal(lastNodes,nodes):
            break
     
    #计算当前K的聚集度
    eva = evaluate_result(center,nodes)             
    #print(eva)


    #展示当前K的聚类效果        
    plotData(nodes,'b','.')
    plotData(center,'r','x')
    
    plt.title('k='+str(K)+",the result of k-means clustering")
    plt.xlabel('x')
    plt.ylabel('y')
    

    plt.show()
    