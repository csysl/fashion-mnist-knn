# -*- coding: utf-8 -*-
"""
    @project:mnist
    @file:train.py
    @ide:PyCharm
    @time:2019-01-05 15:09
    @author:Sun
"""

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from func import LossHistory
from Init import *
import datetime as dt

# compress data
train_images = train_images / 255.0
test_images = test_images / 255.0

# build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# compile the model
model.compile(optimizer=keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 创建损失函数绘图实例
# history = LossHistory()

# train the model
checkpointer = keras.callbacks.ModelCheckpoint(filepath='model/model-{epoch:02d}.h5',
                                               save_best_only=True, verbose=1, period=1)
time_begin = dt.datetime.now()
model.fit(train_images, train_labels,
          validation_data=(test_images, test_labels),
          epochs=epochs,
          batch_size=batch_size,
          callbacks=[keras.callbacks.TensorBoard(log_dir='log'),
                     checkpointer])
time_end = dt.datetime.now()

# # draw the graph of loss and accuracy in training
# history.loss_plot('epoch', epochs)

# print the time of train
total_time = time_end - time_begin
print('The time of train:', total_time, 's')
