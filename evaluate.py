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

# import model
model = keras.models.load_model('model/model' + str(epochs) + '.h5')

# get loss & acc
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('loss:', test_loss)
print('accuracy:', test_acc)
