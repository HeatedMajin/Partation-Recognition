# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:08:26 2017

@author: majin
"""


import numpy as np 
import process_picture as pp
import matplotlib.pyplot as plt

####################################
filein = r'./image/realphoto.jpg'
#filein = r'./image/tests.png'
#filein = r'./image/r2.jpg'
#filein = r'./image/testf.jpg'
#filein = r'./image/testf2.jpg'
####################################
def kernels(nodes,Eps,MinPts):
    '''
    标记所有的点是不是核心点,是nodes[2]=-1,不是nodes[2]=-2
    @nodes
        (n,3) 要标记的点
    @Eps
        范围半径
    @MinPts
        范围内的最少点数
    @return 
        nodes,node_nears
        标记后的点集，以及每个点的临近点
    '''
    node_pisi = np.copy(nodes)
    n = nodes.shape[0]
    nodes_nears = []
    for i in range(n):  #第i个点
        if i%300 ==0:
            print("计算核心点：%.2f%%"%(i/n*100))
        #第i点到其他点的距离np
        #res = np.sum(np.sqrt(np.power(node_pisi[i]-node_pisi,2)),axis=1,keepdims=True)
        res = np.sum(np.abs(node_pisi[i]-node_pisi),axis=1,keepdims=True)
        
        #统计距离小于Eps的点的个数,并记录i点的临近点
        l_nodes = np.less_equal(res,Eps)
        LtEpsCount = np.count_nonzero(l_nodes)
        
        #保存i点的所有临近点
        the_node_nears = np.where(l_nodes)
        #判断是不是核心点
        if LtEpsCount >= MinPts:
            #值保存核心点的邻接点
            nodes_nears.append(the_node_nears[0].tolist())
            nodes[i][2] = -1
        else:
            nodes_nears.append([])
            nodes[i][2] = -2
    return nodes,nodes_nears

def DBScan(nodes,Eps,MinPts):
    '''
    -3代表噪音，-2表示未遍历的非核心，-1表示未遍历的核心点,>0遍历的核心点
    
    @return 
    nodes,(m,3)横纵坐标，类号
    c,分成的类数
    '''
    #标记处所有的核心点
    nodes,nodes_nears = kernels(nodes,Eps,MinPts)
    
    n = nodes.shape[0]
    c = 0
    for i in range(n):
        if i%200 ==0:
            print("开始遍历：%.2f%%"%(i/n*100))
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

def calc_bound(nodes,c):
    '''
    c：被分成的类数
    nodes:(m,3)
    '''
    bound = np.zeros((c,4))#每个类的四个边界,上下左右
    for i in nodes:
        clazz = i[2]
        if clazz<0:  #是噪点不做处理
            continue
        cur_x = i[0]
        cur_y = i[1]
        if clazz>=c or clazz<0:
            continue
            
        bound[clazz][0] = np.max([bound[clazz][0],cur_y])
        bound[clazz][1] = np.min([bound[clazz][1],cur_y])
        if not bound[clazz][1]:#zero
            bound[clazz][1] = cur_y
        bound[clazz][2] = np.min([bound[clazz][2],cur_x])
        if not bound[clazz][2]:#zero
            bound[clazz][2] = cur_x
        bound[clazz][3] = np.max([bound[clazz][3],cur_x])
    return bound
        
        
    
def runner(filein,Eps =5,MinPts=20,show=False):
    '''
    使用DBScan，划分每个像素的类别，并计算每个数字的边界
    -----------
    parameters
        @filein 输入文件的路径
        @Eps DBScan中的个数
        @Minots DBScan中的距离
        @show 是不是展示图片划上切分边框的结果
    '''
    
    img = pp.openImage(filein)
    img_gray = pp.rgb2gray(img)      #变成二值图片
    nodes = load_2img_nodes(img_gray)
    
    #获取图片的名称，使用这个名称保存切割的边界矩阵
    filename = filein.split("/")[-1].split(".")[0]
    try:
        with open("temp_vars/"+str(filename)+".txt",'rb') as f:
            import pickle
            bound = pickle.load(f)
            c = bound.shape[0]
        if show:
            show_res(nodes,c,bound)
        return bound
    except Exception:
        pass
    
    #使用DBScan，划分每个像素的类别
    nodes,c = DBScan(nodes,Eps,MinPts)
    
    #计算每个数字的边界
    bound = calc_bound(nodes,c)
    
    with open("temp_vars/"+str(filename)+".txt",'wb') as f:
        import pickle
        pickle.dump(bound,f)
    
    if show:
        show_res(nodes,c,bound)
    return bound

def show_res(nodes,c,bound):
    '''
    展示聚类和切分结果
    '''
    
    print("总共分成了"+str(c)+"类")
    ################  边框绘制  #####################
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    vertices = []
    codes = []
    for i in bound:            
        max_y = i[0]
        min_y = i[1]
        min_x = i[2]
        max_x = i[3]
        
        codes += [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]
        vertices += [(min_x, max_y), (max_x, max_y), (max_x, min_y), (min_x, min_y),(0, 0)]
    
    vertices = np.array(vertices, float)
    path = Path(vertices, codes)
    
    pathpatch = PathPatch(path, facecolor='None', edgecolor='green')
    
    fig, ax = plt.subplots()
    ax.add_patch(pathpatch)
    ax.set_title('A compound path')
    
    ax.dataLim.update_from_data_xy(vertices)
    ax.autoscale_view()
    
    ################  像素点绘制  #####################
    ax.scatter(nodes[:,0], nodes[:,1],c=nodes[:,2]*30,marker='.')
    
    plt.show()
if __name__ =="__main__":
    runner(filein,show=True)
