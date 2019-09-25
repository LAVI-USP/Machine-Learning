"""
Author: Rodrigo de Barros Vimieiro
Date: August, 2019
rodrigo.vimieiro@gmail.com
=========================================================================

    DESCRIPTION:

    REFERENCE:
     https://ramhiser.com/post/2018-05-14-autoencoders-with-keras/   
        

    Department of Electrical and Computer Engineering, 
    São Carlos School of Engineering, 
    University of São Paulo, 
    São Carlos, Brazil.

    ---------------------------------------------------------------------
    Copyright (C) <2018>  <Rodrigo de Barros Vimieiro>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.    
 

=========================================================================
"""

#%% Imports

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose

#%% Data preprocess
def data_preprocess(x_train, y_train, x_test, y_test, param):

    # Converts a class vector (integers) to binary class matrix.
    # One-hot encoding
    y_train_out = tf.keras.utils.to_categorical(y_train,param['num_classes'],dtype='uint8')
    y_test_out = tf.keras.utils.to_categorical(y_test,param['num_classes'],dtype='uint8')

    # Normalize data (0-255 -> 0-1)
    x_train_out =  x_train / 255
    x_test_out =  x_test / 255

    # Reshape data to 4 dimensions (Num_img_train, Img_rows, Img_cols, Img_depth(channels) -> x_train[0,:,:,0] -> Img 1)
    x_train_out = x_train_out.reshape(param['num_images_train'], param['img_rows'], param['img_cols'], 1)
    x_test_out = x_test_out.reshape(param['num_images_test'], param['img_rows'], param['img_cols'], 1)

    return (x_train_out, y_train_out), (x_test_out, y_test_out)

#%% Noise injection
def noise_injection(data,mean,std):

    noise_data = data + np.random.normal(loc= mean, scale= std, size=data.shape)
    noise_data = np.clip(noise_data, 0., 1.)

    return noise_data

#%% Data show
def data_show(x_test,prediction,nimages):

    nrandom = np.random.choice(prediction.shape[0],nimages)

    figure = plt.figure(figsize=(14, 6))

    for i, index in enumerate(nrandom):
        subplot = figure.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
        subplot.imshow(np.squeeze(x_test[index]),'gray')

        subplot = figure.add_subplot(2, 5, (i + 1) + nimages, xticks=[], yticks=[])
        subplot.imshow(np.squeeze(prediction[index]),'gray')
    plt.show()
    return

#%% **************************** Main Code ************************************

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Create a dictionary with general parameters of the dataset
param = {'num_classes': len(np.unique(y_train)),
         'img_rows': x_train.shape[1],
         'img_cols': x_train.shape[2],
         'num_images_train': x_train.shape[0],
         'num_images_test': x_test.shape[0]
         }

# Pre-process function
(x_train, y_train), (x_test, y_test) = data_preprocess(x_train,y_train,x_test,y_test,param)

# Inject AWG I.I.D noise: Mean x and std y
x_train_noise = noise_injection(x_train,mean=0,std=0.4)
x_test_noise = noise_injection(x_test,mean=0,std=0.4)

#%%
# Create the network structure (Convolutional autoencoder)

model = Sequential()

# Encoder Layers

# Convolutional layer: convolution + activation + pooling #
# Convolution: kernel: 3x3x32 (32 filters); stride: 1; padding: 'same', i.e., 'replicate'; Input shape: 28x28x1
# Activation function: 'relu'
model.add(Conv2D(32, kernel_size=(3, 3),
                 strides=1,
                 padding='same',
                 activation='relu',
                 input_shape=(param['img_rows'], param['img_cols'], 1)))
# Pooling: Downsampling with max value (stride 2x2)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, kernel_size=(3, 3),
                 strides=1,
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten encoding for visualization
# model.add(Flatten())

#model.add(Dropout(0.25))
#model.add(Dense(16, activation='relu'))

# Decoder Layers
#model.add(Dense(576, activation='relu'))
#model.add(Reshape((5, 5, 16)))
#model.add(UpSampling2D((2,2)))

# Scaling up of the image (2x2) -> Interpolation
#model.add(UpSampling2D((2,2)))
# Transposed Convolutions (It is not deconvolution, because deconvolution is the mathematical inverse of convolution)
# "... We can combine the upscaling of an image with a convolution, instead of doing two separate processes"
# https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d
# https://medium.com/apache-mxnet/transposed-convolutions-explained-with-ms-excel-52d13030c7e8
# https://stackoverflow.com/questions/53654310/what-is-the-difference-between-upsampling2d-and-conv2dtranspose-functions-in-ker
model.add(Conv2DTranspose(16, kernel_size=(3,3),
                          strides=2,
                          padding='same',
                          activation='relu'))
#model.add(UpSampling2D((2,2)))
model.add(Conv2DTranspose(32, kernel_size=(3,3),
                          strides=2,
                          padding='same',
                          activation='relu'))

model.add(Conv2DTranspose(1, kernel_size=(3,3),
                          strides=1,
                          padding='same',
                          activation='relu'))

# Show model summary
model.summary()

#%%
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')

# Train the network
model.fit(x_train_noise, x_train,
                epochs= 2,
                batch_size= 128,
                validation_split = 0.2)

#%%

decoded_imgs = model.predict(x_test_noise)

# Show some decoded images
data_show(x_test_noise, decoded_imgs, nimages=5)