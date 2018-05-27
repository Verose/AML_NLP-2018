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

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0


x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)

z_log_var = Dense(latent_dim, trainable=False,kernel_initializer='zeros',bias_initializer='zeros')(h)



def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

encoder = Model(x, z_mean)
encoder.summary()

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(inputs=decoder_input, outputs= _x_decoded_mean)


# instantiate VAE model
vae = Model(x, x_decoded_mean)

# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
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

    z_mean = encoder.predict(np.asarray(test_samples_by_digit))
    colors = cm.rainbow(np.linspace(0, 1, len(test_samples_by_digit)))
    fig, ax = plt.subplots()
    ax.scatter([point[0] for point in z_mean], [point[1] for point in z_mean], color=colors)

    for idx in range(10):
        ax.annotate(idx, (z_mean[idx][0],z_mean[idx][1]))
    fig.savefig("latent_images_const_variance")
    table_txt = str([point[0] for point in z_mean]) + '\n' + str([point[1] for point in z_mean])
    text_file = open("z_mean_fixed_var_table.txt", "w")
    text_file.write(table_txt)
    text_file.close()

    z_sample = np.array([[0.5, 0.2]])
    x_decoded_sample = generator.predict(z_sample)
    x_decoded_sample = x_decoded_sample.reshape((28,28))
    plt.imsave("x_decoded_sample_const_variance" + '.png', x_decoded_sample)


    first_img = x_test[0]
    first_label = y_test[0]
    first_img_latent_repr = encoder.predict(np.asarray([np.asarray(first_img)]))
    
    second_img = x_test[1]
    second_label = y_test[1]
    second_img_latent_repr = encoder.predict(np.asarray([np.asarray(second_img)]))

    x_progress_first_second = np.linspace(first_img_latent_repr[0][0],second_img_latent_repr[0][0],10)
    y_progress_first_second = np.linspace(first_img_latent_repr[0][1],second_img_latent_repr[0][1],10)

    for image_progress_idx in range(len(x_progress_first_second)):
        image_progress = generator.predict(np.asarray([np.asarray((x_progress_first_second[image_progress_idx],y_progress_first_second[image_progress_idx]))]))
        image_progress = image_progress.reshape((28,28))

        plt.imsave("image_progress_const_variance__" + str(image_progress_idx) + '.png', image_progress)