# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 16:38:34 2017

@author: majin
"""
import numpy as np
def kernels(nodes,Eps,MinPts):
    '''
    标记所有的点是不是核心点,是nodes[2]=-1,不是nodes[2]=-2
    @nodes
        (n,3) 要标记的点
    @Eps
    @MinPts
        范围内的最少点数
    @return 
        nodes,node_nears
        标记后的点集，以及每个点的临近点
    '''
    node_pisi = np.copy(nodes)
    n = nodes.shape[0]
    node_nears = []
    for i in range(n):  #第i个点
        the_nears = []
        #第i点到其他点的距离
        #print(res)
        res = np.sqrt(np.power(node_pisi[i]-node_pisi,2))
        res = np.sum(res,axis=1,keepdims=True)
        
        #统计距离小于Eps的点的个数,并记录i点的临近点
        LtEpsCount = 0
        for ii in range(n-1):
            if res[ii][0]<=Eps :
                LtEpsCount += 1
                the_nears.append(ii)
        
        #保存i点的所有临近点
        node_nears.append(the_nears)
        #判断是不是核心点
        if LtEpsCount >= MinPts:
            nodes[i][2] = -1
        else:
            nodes[i][2] = -2
    #print(nodes)
    return nodes,node_nears

def DBScan(nodes,Eps,MinPts):
    '''
    -3代表噪音，-2表示未遍历的非核心，-1表示未遍历的核心点,>0遍历的核心点
    '''
    #标记处所有的核心点
    nodes,nodes_nears = kernels(nodes,Eps,MinPts)
    
    n = nodes.shape[0]
    c = 0
    for i in range(n):
        print(str(round(i/n*100,2))+"%")
        if nodes[i][2]==-2:#未遍历的非核心
            nodes[i][2] = -3#-3代表噪音
            continue
        elif nodes[i][2]==-1:#是未遍历的核心点
            nodes[i][2] = c#标记P为c
            P_near = nodes_nears[i] #取出P的临近点
            for ii in P_near:
                ii = int(ii)
                if nodes[ii][2] == -2 or nodes[ii][2]==-3:#是噪声或未遍历的非核心
                    nodes[ii][2] = c
                elif nodes[ii][2] == -1:#是未遍历的核心点
                    nodes[ii][2] = c
                    #将P'的临近点加入到M
                    for shouldExt in nodes_nears[ii]:
                        if shouldExt not in P_near:
                            P_near.append(shouldExt)
            c += 1
    return nodes,c
    
    