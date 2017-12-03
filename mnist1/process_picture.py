# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 20:15:20 2017

@author: majin
"""
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import DBScan_segment as dbss

def getTestPicArray(filename) :
    im = Image.open(filename)
    x_s = 28
    y_s = 28
    out = im.resize((x_s, y_s), Image.ANTIALIAS) 
        
    im_arr = np.array(out.convert('L'))
	
    num0 = 0
    num255 = 0
    threshold = 100

    for x in range(x_s):
        for y in range(y_s):
            if im_arr[x][y] > threshold : num255 = num255 + 1
            else : num0 = num0 + 1

    if(num255 > num0) :
        print("convert!")
        for x in range(x_s):
            for y in range(y_s):
                im_arr[x][y] = 255 - im_arr[x][y]
                if(im_arr[x][y] < threshold) :  im_arr[x][y] = 0
    '''
    out = Image.fromarray(np.uint8(im_arr))
    
    fname = filename.split("/")[-1].split(".")[0]
    imType = filename.split("/")[-1].split(".")[1]
    out.save(fname+"."+imType)
    ''' 
    nm = im_arr.reshape((1, 784))
	
    nm = nm.astype(np.float32)
    nm = np.multiply(nm, 1.0 / 255.0)
    return nm


def openImage(fileImg):
    return Image.open(fileImg)

def resizeImage(img,  width, height, imtype="png",fileout=None):
    '''
    filein: 输入图片
    fileout: 输出图片
    width: 输出图片宽度
    height:输出图片高度
    type:输出图片类型（png, gif, jpeg...）
    '''
    
    #resize image with high-quality
    out = img.resize((width, height),Image.ANTIALIAS) 
    if fileout:
        out.save(fileout, imtype)
    return out
def rgb2gray(pic,show=False): 
    '''
     Y' = 0.299 R + 0.587 G + 0.114 B
     
     L模式：灰度模式
     1模式：二值模式
    '''
    #将灰度值
    pic = pic.convert("L")
    #tva = [ (255-x)*1.0/255.0 for x in pic.getdata()] 
    for i in range(pic.width):
        for j in range(pic.height):
            h = pic.getpixel((i,j))
            if h>110:
                h=255   #设成白色
            else:
                h=0     #设成黑色
            pic.putpixel((i,j), h) 
    if show:
        pic.show()    
    return pic
def genPixValue(pic,show=False): 
    im_arr = np.array(pic.convert('L'))
    threshold = 100

    x_s = 28
    y_s = 28

    for x in range(x_s):
        for y in range(y_s):
            im_arr[x][y] = 255 - im_arr[x][y]
            if(im_arr[x][y] < threshold) :  im_arr[x][y] = 0
    
    nm = im_arr.astype(np.float32)
    nm = np.multiply(nm, 1.0 / 255.0)
    return nm.reshape((1, 784))

def segment_picture(filein,blank_padding = 10):
    '''
    切分图片
    ---------------
    paramters
        @filein：图像的路径
        @blannk_padding: 图像切分时，保留多少的padding
    ---------------
    return
        @[0] : 返回切分的图像矩阵,维度(c,784)
                c:类数，聚了几类切成几类
                784：切下来的图片，resize
    '''
      
    #使用DBScan聚类来分割图片
    #DBScan后，得到类的边界（使用路径参数的原因，为了序列化）
    bound = dbss.runner(filein) 
    
    pic = openImage(filein)  #打开图片
    pic = pic.convert("L")
    
    imgmatric = []           #保存切出来的图片
                             #(c,784),每行是一个数字，784是选框确定后resize的结果

    for i in bound:
        height = pic.height
        
        #数字上下左右的边界
        max_y = i[0]
        min_y = i[1]
        min_x = i[2]
        max_x = i[3]
        
        #图像的左上
        x1 = min_x-blank_padding
        y1 = height-max_y-blank_padding
        
        #图像的右下
        x2 = max_x+blank_padding
        y2 = height-min_y+blank_padding
        
        #切分图片
        sp_im = pic.crop((x1,y1,x2,y2))
        
        #保存的图像是28x28的
        resize_sp_im = resizeImage(sp_im,28,28)  
        
        resize_sp_im = genPixValue(resize_sp_im)
        #转成行向量
        

        #矩阵添加一行
        imgmatric.append(resize_sp_im[0])
    return np.matrix(imgmatric)



if __name__ == "__main__":
   
    #filein = r'./image/r2.jpg'
    filein = r'./image/realphoto.jpg'
    #filein = r'./image/tests.png'
    #filein = r'./image/testf.jpg'
    
    
    #测试去掉真实世界的背景
    a = openImage(filein)
    rgb2gray(a,show=True)
    '''

    
    #测试分割图像    
    res = segment_picture(filein,blank_padding=3)
    clazz_num = res.shape[0]
    for i in range(clazz_num):
        plt.subplot(clazz_num/4+1,4,i+1)
        plt.imshow(res[i].reshape([28,28]))
    
    '''