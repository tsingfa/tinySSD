# -*- coding: utf-8 -*-

"""
@Project : tinySSD
@FileName: configs.py
@Software: PyCharm

@Time    : 2022/10/25 3:00
@Author  : 青fa
@Email   : tsingfa6@gmail.com
@GitHub  : https://github.com/tsingfa

功能描述:统一变量参数指定

"""



#----------------------------model参数----------------------------#

model_zoo=['base','resnet']     #模型选择
model_train=model_zoo[0]        #选择模型进行训练
model_test = model_zoo[0]



#----------------------------train相关参数----------------------------#



epoch_train=50  #训练轮数
batch_size=32
lr=0.2
weight_decay=5e-4
train_msg=model_train+'+SGD+CosineAnnealingLR+'+str(epoch_train)    #与图片、权重的命名和保存相关




#----------------------------test相关参数----------------------------#

epoch_test = 50 # 调用第几个epoch的模型结果进行测试
threshold = 0.3  # 置信阈值



#----------------------------data相关参数----------------------------#




