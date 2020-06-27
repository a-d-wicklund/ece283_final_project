'''Example of VAE on MNIST dataset using MLP

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean = 0 and std = 1.

# Reference

[1] Kingma, Diederik P., and Max Welling.
"Auto-Encoding Variational Bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
from matplotlib import pyplot as plt
import argparse
import os
from datetime import datetime

import midi
import load_songs

os.environ["CUDA_VISIBLE_DEVICES"]="2"

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def save_models(models):
    encoder, decoder, vae = models
    model_encoder_json = encoder.to_json()
    with open("archived/model_encoder.json", "w") as json_file:
        json_file.write(model_encoder_json)
    model_decoder_json = decoder.to_json()
    with open('archived/model_decoder.json', 'w') as json_file:
        json_file.write(model_decoder_json)
    model_vae_json = vae.to_json()
    with open('archived/model_vae.json', 'w') as json_file:
        json_file.write(model_vae_json) 


def save_training_history(history):
    np.save('archived/val_loss.npy', history.history['val_loss'])
    np.save('archived/loss.npy', history.history['loss']) 



def create_samples(models, x_test, latent_dim):
    variance_mult = 2
    encoder, decoder = models
    z_mean, z_log_var, z = encoder.predict(x_test, batch_size=x_test.shape[0])
    epsilon = np.random.normal(size=(x_test.shape[0], latent_dim))
    latent_vec = z_mean + variance_mult * np.exp(0.5 * z_log_var) * epsilon
    samples_out = decoder.predict(latent_vec, batch_size=x_test.shape[0])
    # Save mean histogram
    np.save('z_mean_smallbeta.npy', z_mean)
    #plt.hist(z_mean)
    #plt.title('Histogram of latent space mean: beta = 5')
    #plt.xlabel('Mean produced in latent space')
    return samples_out


# Grab data
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
x_all = load_songs.main(from_file=True)
#x_all = np.load(samples.npy)
# Use 90% of the data for training and the other 10% for testing
x_train = x_all[:x_all.shape[0]*9//10]
np.random.shuffle(x_train) # Shuffle so that each batch has variety
x_test = x_all[x_all.shape[0]*9//10:]



original_dim = x_train.shape[1] * x_train.shape[2]
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
#x_train = x_train.astype('float32') / 255
#x_test = x_test.astype('float32') / 255

# network parameters
input_shape = (original_dim, )
intermediate_dim = 1024
batch_size = 256
latent_dim = 2
epochs = 50
beta = 5 #multiplier for kl divergence

# VAE model = encoder + decoder
# build encoder model
# For midi application, take
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    save_models((encoder, decoder, vae))
    #data = (x_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss)
    #vae_loss = K.mean(reconstruction_loss + beta * kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    plot_model(vae,
               to_file='vae_mlp.png',
               show_shapes=True)

    if args.weights:
        vae.load_weights(args.weights)
    else:
        # train the autoencoder
        history = vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))
        vae.save_weights('vae_mlp_mnist.h5')
        save_training_history(history)	
    
    # Grab example output from the autoencoder
    sample_test = x_test[2000:2100]
    #x_decoded = vae.predict(sample_test, batch_size=100)
    #midi.samples_to_midi([np.reshape(x_decoded, [-1, 96, 96])[me] for me in range(10)], '../nn_output/vae_output_' + datetime.now().strftime('%H%M') + '.mid')
    #midi.samples_to_midi([np.reshape(sample_test, [-1, 96, 96])[me] for me in range(10)], '../nn_output/vae_input_' + datetime.now().strftime('%H%M') + '.mid')
    samples_out = create_samples((encoder, decoder), x_test[1000:10000], latent_dim=latent_dim) 
    
    #save_results(models, x_test, batch) 

#    plot_results(models,
#                 data,
#                 batch_size=batch_size,
#                 model_name="vae_midi")
