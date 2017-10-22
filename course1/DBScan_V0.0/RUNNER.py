# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 16:31:36 2017

@author: majin
"""
import DBScan
import Utils
import matplotlib.pyplot as plt
if __name__ =="__main__":        
    filePath = "Aggregation.txt"
    #读取文件
    initNodes = Utils.readFile2Matrix(filePath) #(n,3)
    
    nodes,c = DBScan.DBScan(initNodes,1.6,6)
    
    print(nodes[0:20])
    print("总共分成了"+str(c)+"类")
    Utils.plotData(nodes,'g','.')
    plt.title("the view of clustering")
    plt.xlabel('x')
    plt.ylabel('y')
    Utils.show()
    