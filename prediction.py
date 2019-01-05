# -*- coding: utf-8 -*-
"""
    @project:mnist
    @file:prediction.py
    @ide:PyCharm
    @time:2019-01-05 16:09
    @author:Sun
"""
from tensorflow import keras
from Init import *

# load model
model = keras.models.load_model('model/model' + str(epochs) + '.h5')

# predictions shape (10000,10)
predictions = model.predict(test_images)

print(predictions[0])
print(predictions.shape)
