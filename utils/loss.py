# -*- coding: utf-8 -*-

"""
@Project : tinySSD
@FileName: loss.py
@Software: PyCharm

@Time    : 2022/10/25 1:30
@Author  : 青fa
@Email   : tsingfa6@gmail.com
@GitHub  : https://github.com/tsingfa

功能描述：损失函数计算

"""
import torch.nn as nn




def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls_loss = nn.CrossEntropyLoss(reduction='none')
    bbox_loss = nn.L1Loss(reduction='none')

    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox