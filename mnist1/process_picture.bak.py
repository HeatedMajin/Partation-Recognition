# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 09:36:19 2017

@author: majin
"""
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import DBScan_segment as dbss

def get_my_img_1x784(img_path, show=True):
    '''
    获取输入的单数字图片
    '''
    #获取输入图片  将输入图片大小调整成28x28
    img_28x28 = resizeImage(img_path,28,28)
    gray_img_28x28 = rgb2gray(img_28x28)
    if show:
        plt.imshow(gray_img_28x28)
        plt.show()
  
    #将图片转成矩阵，feed需要1x784
    img_1x784 = np.matrix(gray_img_28x28.getdata()).reshape([1,-1])
    return img_1x784

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
            if h>200:
                h=255   #设成白色
            pic.putpixel((i,j), h) 
    if show:
        pic.show()    
    return pic


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
    pic = rgb2gray(pic)      #转成纯灰度图像    
    
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
        
        #转成行向量
        img_1x784 = np.matrix(resize_sp_im.getdata()).reshape([1,-1])
        
        #矩阵添加一行
        imgmatric.append(img_1x784[0].tolist()[0])
    return np.matrix(imgmatric)



if __name__ == "__main__":
   
    #filein = r'./image/r2.jpg'
    filein = r'./image/realphoto.jpg'

    
    '''
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
    
    