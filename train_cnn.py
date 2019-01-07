# -*- coding: utf-8 -*-
"""
    @project:mnist
    @file:train_cnn.py
    @ide:PyCharm
    @time:2019-01-05 21:28
    @author:Sun
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
from Init import *
import datetime as dt

# compress data
train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)
train_images = train_images[..., np.newaxis]
test_images = test_images[..., np.newaxis]

# build the model
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=64, kernel_size=(11, 11), strides=(1, 1), padding='valid',
                              activation=tf.keras.activations.relu, input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(keras.layers.Conv2D(filters=192, kernel_size=(5, 5), strides=(1, 1), padding='same',
                              activation=tf.keras.activations.relu))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same',
                              activation=tf.keras.activations.relu))
model.add(keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same',
                              activation=tf.keras.activations.relu))
model.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                              activation=tf.keras.activations.relu))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=4096, activation=keras.activations.relu))
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(units=4096, activation=keras.activations.relu))
model.add(keras.layers.Dropout(rate=0.5))

model.add(keras.layers.Dense(units=10, activation=keras.activations.softmax))

# compile the model
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.categorical_crossentropy,
              # loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model
checkpointer = keras.callbacks.ModelCheckpoint(filepath='model2/model_cnn-{epoch:02d}.h5',
                                               save_best_only=True, verbose=1, period=1)
time_begin = dt.datetime.now()
model.fit(train_images, train_labels,
          validation_data=(test_images, test_labels),
          epochs=epochs,
          batch_size=batch_size,
          callbacks=[keras.callbacks.TensorBoard(log_dir='log'),
                     checkpointer])
time_end = dt.datetime.now()

# print the time of train
total_time = time_end - time_begin
print('The time of train:', total_time, 's')

# # save the model
# model.save('model/model_cnn' + str(epochs) + '.h5')
