#!/usr/bin/env python
# coding: utf-8

# predict.py

"""
The script predicts both the class and the certainty for any designated image. Since the 
prediction-related function is a customized relaization. It is quire different from the 
decode_predictions() within keras. However, the latter only accepts 1000 classes not 1001 t
hat is defaulted in the Inception V4 Weights. Please give the commands as follows. 

$ python predict.py

Class is: African elephant, Loxodonta africana
Certainty is: 0.8177135

Please pay more attention on the formal argument "x". To faciliate the process of parameter passing
during the function calls in the context, we select x to express the recursion that is the typical
mathematical usage. 
"""

import tensorflow as tf 
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from densenet_func import DenseNet169

# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def preprocess_input(x):
    
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    output = np.multiply(x, 2.0)

    return output


if __name__ == '__main__':


    model = DenseNet169(include_top=True, weights='imagenet')

    model.summary()

    img_path = '/home/mike/Documents/keras_densenet/elephant.jpg'
    img = image.load_img(img_path, target_size=(224,224))
    output = preprocess_input(img)
    print('Input image shape:', output.shape)

    preds = model.predict(output)
    print(np.argmax(preds))
    print('Predicted:', decode_predictions(preds,1))
