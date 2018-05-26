'''This script demonstrates how to build a variational autoencoder with Keras.

 #Reference

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.cm as cm

from keras.layers import Input, Dense, Lambda, Reshape, Flatten, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

batch_size = 10
original_dim = 784
latent_dim = 2
intermediate_dim = 16
epochs = 0
epsilon_std = 1.0


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon
# Definition of Keras ConvNet architecture

input_shape = (28, 28, 1)
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
x = Conv2D(16, (3, 3), activation='relu', padding='same', name='Conv2D1')(x)
x = MaxPooling2D((2, 2), padding='same', name='MaxPooling2D1')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same', name='Conv2D2')(x)
x = MaxPooling2D((2, 2), padding='same', name='MaxPooling2D2')(x)
# shape info needed to build decoder model
shape = K.int_shape(x) 
# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(intermediate_dim, activation='relu', name='Dense1')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

encoder = Model(inputs, [z_mean, z_log_var, z])
pre_decoder_inputs = Input(shape=(latent_dim,))

x = Dense(shape[1] * shape[2] * shape[3], activation='relu', name='Dense4')(pre_decoder_inputs)
x = Reshape((shape[1], shape[2], shape[3]), name='Reshape')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same', name='Conv2D3')(x)
x = UpSampling2D((2, 2), name='UpSampling2D1')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same', name='Conv2D4')(x)
x = UpSampling2D((2, 2), name='UpSampling2D2')(x)
outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='Conv2D5')(x)

decoder = Model(pre_decoder_inputs, outputs, name='decoder')

vae = Model(inputs, outputs, name='vae')


# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(x, outputs)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()


# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))

if __name__ == '__main__':

    test_samples_by_digit = []
    cur_digit = 0
    for idx,label in enumerate(y_test):
        if(label == cur_digit):
            cur_digit += 1
            test_samples_by_digit.append(x_test[idx])
        if(cur_digit == 10):
            break

    z_mean, _, _ = encoder.predict(np.asarray(test_samples_by_digit))
    colors = cm.rainbow(np.linspace(0, 1, len(test_samples_by_digit)))
    fig, ax = plt.subplots()
    ax.scatter([point[0] for point in z_mean], [point[1] for point in z_mean], color=colors)

    for idx in range(10):
        ax.annotate(idx, (z_mean[idx][0],z_mean[idx][1]))
    fig.savefig("latent_images_convnet")

    first_img = x_test[0]
    first_label = y_test[0]
    first_img_latent_repr, _, _ = encoder.predict(np.asarray([np.asarray(first_img)]))
    
    second_img = x_test[1]
    second_label = y_test[1]
    second_img_latent_repr, _, _ = encoder.predict(np.asarray([np.asarray(second_img)]))

    x_progress_first_second = np.linspace(first_img_latent_repr[0][0],second_img_latent_repr[0][0],10)
    y_progress_first_second = np.linspace(first_img_latent_repr[0][1],second_img_latent_repr[0][1],10)

    for image_progress_idx in range(len(x_progress_first_second)):
        image_progress = generator.predict(np.asarray([np.asarray((x_progress_first_second[image_progress_idx],y_progress_first_second[image_progress_idx]))]))
        image_progress = image_progress.reshape((28,28))

        plt.imsave("image_progress_convnet__" + str(image_progress_idx) + '.png', image_progress)






