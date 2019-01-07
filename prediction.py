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
model = keras.models.load_model('model/model_cnn-10.h5')

# predictions: shape (10000,10)
test_images2 = test_images
test_images = test_images / 255.0
test_images = test_images[..., np.newaxis]
predictions = model.predict(test_images)

if __name__ == '__main__':

    tno = [1,21,5,19,555,355,3333,4444,8888,9999]
    for no in tno:
        img=test_images[no,...]
        # prediction
        img2=img[np.newaxis,...]
        img_pred=model.predict(img2)
        #print(img_pred[0])
        # visualization
        plt.figure()
        plt.imshow(np.squeeze(img))
        plt.colorbar()
        plt.grid(False)
        # get the result of prediction
        pred_c=img_pred[0].tolist()
        pred_category = class_names[pred_c.index(max(pred_c))]
        real_category = class_names[test_labels[no]]
        plt.title('Predict category:' + pred_category +
                  '&Real category:' + real_category)
        plt.savefig('result/result'+str(no)+'.png')
        plt.show()

# img=test_images[2,...]
# # prediction
# img2=img[np.newaxis,...]
# img_pred=model.predict(img2)
# pred_category = class_names[predictions[no].tolist().index(1)]
# real_category = class_names[test_labels[no]]

# print(predictions[1])
# print(predictions[0].tolist().index(1))
# print(predictions.shape)
