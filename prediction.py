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
import matplotlib.pyplot as plt
import numpy as np

# load model
model = keras.models.load_model('model/model' + str(epochs) + '.h5')

# predictions: shape (10000,10)
predictions = model.predict(test_images)


if __name__=='__main__':
    no=int(input("please input the picture's no in test dataset(0-9999):"))
    plt.figure()
    plt.imshow(test_images[no])
    plt.colorbar()
    plt.grid(False)
    # get the result of prediction
    category=class_names[predictions[0].tolist().index(1)]
    plt.title('the category:'+category)
    plt.show()

# print(predictions[0].tolist().index(1))
# print(predictions.shape)
