# -*- coding: utf-8 -*-
"""
    @project:mnist
    @file:Init.py
    @ide:PyCharm
    @time:2019-01-05 16:19
    @author:Sun
"""
from func import Data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# todo:data prepare
# DataPrepare()

# get data
train_images, train_labels = Data.GetTrainData()
test_images, test_labels = Data.GetTestData()

epochs = 10
batch_size = 20
