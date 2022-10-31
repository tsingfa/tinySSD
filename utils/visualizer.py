# -*- coding: utf-8 -*-

"""
@Project : tinySSD
@FileName: visualizer.py
@Software: PyCharm

@Time    : 2022/10/21 1:31
@Author  : 青fa
@Email   : tsingfa6@gmail.com
@GitHub  : https://github.com/tsingfa

功能描述：可视化展示测试结果

"""
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from configs import *

def bbox_to_rect(bbox, color):
    # 将边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式：
    # ((左上x,左上y),宽,高)
    return patches.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
        fill=False, edgecolor=color, linewidth=2)

def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示所有边界框"""

    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.detach().cpu().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))

def display(img, path,output, threshold ):
    fig = plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
    plt.savefig(path)
    plt.show()

#解决中文画图的乱码问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

def train_plot(cls_err_lst,bbox_mae_lst):
    plt.plot(cls_err_lst,label="class error")
    plt.plot(bbox_mae_lst,label="bbox mae")
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title("训练过程class error与bbox mae变化图")

    train_fig_path='saved_figures/train/cls_err+bbox_mae'
    if not os.path.exists(train_fig_path):
        os.makedirs(train_fig_path)
    plt.savefig(train_fig_path+'/'+train_msg)
    plt.show()