# -*- coding: utf-8 -*-
"""
    @project:mnist
    @file:train_cnn2.py
    @ide:PyCharm
    @time:2019-01-07 17:29
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

# # load model
# model = keras.models.load_model('model/model_cnn-10.h5')

model = tf.keras.Sequential()

# Must definethe input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same',
                                 activation=keras.activations.relu, input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same',
                                 activation=keras.activations.relu))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation=keras.activations.relu))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation=keras.activations.softmax))

# compile the model
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.categorical_crossentropy,
              # loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model
checkpointer = keras.callbacks.ModelCheckpoint(filepath='model3/model_cnn-{epoch:02d}.h5',
                                               save_best_only=True, verbose=1, period=1)
time_begin = dt.datetime.now()
model.fit(train_images, train_labels,
          validation_data=(test_images, test_labels),
          epochs=epochs,
          batch_size=batch_size,
          # initial_epoch=10,
          shuffle=True,  # shuffle the train data before each epoch
          callbacks=[keras.callbacks.TensorBoard(log_dir='log3'),
                     checkpointer])
time_end = dt.datetime.now()

# print the time of train
total_time = time_end - time_begin
print('The time of train:', total_time, 's')

# # save the model
# model.save('model/model_cnn' + str(epochs) + '.h5')
