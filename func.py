# -*- coding: utf-8 -*-
"""
    @project:mnist
    @file:func.py
    @ide:PyCharm
    @time:2019-01-05 15:40
    @author:Sun
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np


# 绘制训练/测试集的某张图片
def DrawPicture(picture=None):
    '''
    :param picture: shape(28,28)
    :return: a picture
    '''
    plt.figure()
    plt.imshow(picture)
    plt.colorbar()
    plt.grid(False)
    plt.show()

# todo:数据准备
class Data():
    def DataPrepare(self):
        self.fashion_mnist = keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) \
            = fashion_mnist.load_data()
        np.save('dataset/train_images.npy', self.train_images)
        np.save('dataset/train_labels.npy', self.train_labels)
        np.save('dataset/test_images.npy', self.test_images)
        np.save('dataset/test_labels.npy', self.test_labels)

    def GetTrainData():
        train_images = np.load('dataset/train_images.npy')
        train_labels = np.load('dataset/train_labels.npy')
        return train_images, train_labels

    def GetTestData():
        test_images = np.load('dataset/test_images.npy')
        test_labels = np.load('dataset/test_labels.npy')
        return test_images, test_labels

# todo:绘制训练的loss&accuracy
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type, epochs):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='test acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='test loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend()
        plt.title("acc-loss", fontsize=20)
        plt.savefig('result/acc-loss' + str(epochs) + '.png', dpi=1000)
        plt.show()
