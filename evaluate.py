# -*- coding: utf-8 -*-
"""
    @project:mnist
    @file:evaluate.py
    @ide:PyCharm
    @time:2019-01-05 19:08
    @author:Sun
"""
from tensorflow import keras
from Init import *
import numpy as np

# import model
model = keras.models.load_model('model/model_cnn-10.h5')

# show the construction of the model
# model.summary()

# get loss & acc
test_images=test_images/255.0
test_images = test_images[..., np.newaxis]
test_labels = keras.utils.to_categorical(test_labels, 10)
test_loss, test_acc = model.evaluate(test_images, test_labels)
# print(model.evaluate(test_images, test_labels))

# print the accuracy of the test dataset
print('accuracy:{:5.2f}%'.format(100*test_acc))
