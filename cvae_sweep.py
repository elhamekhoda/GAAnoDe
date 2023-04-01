import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Input, Dense, Add, Activation, Concatenate
from tensorflow.keras.callbacks import History, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Concatenate, LeakyReLU, Reshape, Conv2DTranspose, Flatten, Conv2D
from tensorflow.keras.models import Model,Sequential

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint, WandbCallback

from utils import *

#setup wandb configs
configs = dict(
        learning_rate = 0.001, 
        beta = 10**-2,
        batch_size = 512,
        latent_dim = 6,
        epochs = 100
)

sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'epoch_loss'},
    'parameters': 
    {
        'learning_rate': {'max': 0.003, 'min': 0.0005},
        'beta': {'max': 0.2, 'min': 0.005}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project='anomoly_detection')

#setup GPU env
disable_eager_execution()

physical_devices = tf.config.experimental.list_physical_devices()
print("All available physical devices:", physical_devices)

# Select a GPU device for training
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    device = gpu_devices[0]
    tf.config.experimental.set_memory_growth(device, True)
    tf.config.experimental.set_visible_devices(device, 'GPU')
    print("Selected GPU device:", device)
else:
    print("No GPU devices found.")
    
#prepare training data
outerdata_train = np.load("./data/preprocessed_data_6var_more_training_data/outerdata_train_6var.npy")
outerdata_test = np.load("./data/preprocessed_data_6var_more_training_data/outerdata_test_6var.npy")
nFeat = 6
input_dim = 6
outerdata_train = outerdata_train[outerdata_train[:,nFeat+1]==0]
outerdata_test = outerdata_test[outerdata_test[:,nFeat+1]==0]
data_train = outerdata_train[:,1:nFeat+1]
data_test = outerdata_test[:,1:nFeat+1]
data = np.concatenate((data_train, data_test), axis=0)
cond_data_train = outerdata_train[:,0]
cond_data_test = outerdata_test[:,0]
cond_data = np.concatenate((cond_data_train, cond_data_test), axis=0)
#normalization
data, data_max, data_min = minmax_norm_data(data)
cond_data, cond_data_max, cond_data_min  = minmax_norm_cond_data(cond_data)
data = logit_norm(data)
cond_data = logit_norm(cond_data)
#more data processing 
trainsize = outerdata_train.shape[0]
data = data[:,0:input_dim]
data = np.reshape(data, (len(cond_data),input_dim))
x_train = data[:trainsize]
x_test = data[trainsize:]
y_train = cond_data[:trainsize]
y_test = cond_data[trainsize:]
image_size = x_train.shape[1]
original_dim = image_size
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = np.reshape(y_train, [-1, 1])
y_test = np.reshape(y_test, [-1, 1])
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

#define saving folder
folder_name = "cvae_sweep/"
comd = "mkdir -p "+"./outputs/models/"+folder_name
os.system(comd)

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def create_encoder(X,y, latent_dim):
    inputs = Concatenate()([X, y])
    x1 = Dense(32, activation='relu')(inputs)
    x2 = Dense(128, activation='relu')(x1)
    x3 = Dense(128, activation='relu')(x2)
    x4 = Dense(32, activation='relu')(x3)
    z_mean = Dense(latent_dim, name='z_mean')(x4)
    z_log_var = Dense(latent_dim, name='z_log_var')(x4)
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    return Model([X, y], [z_mean, z_log_var, z], name='encoder')

def create_decoder(z,y):
    latent_inputs = Concatenate()([z, y])
    x1 = Dense(32, activation='relu')(latent_inputs)
    x2 = Dense(128, activation='relu')(x1)
    x3 = Dense(128, activation='relu')(x2)
    x4 = Dense(32, activation='relu')(x3)

    outputs = Dense(input_dim, activation='linear')(x4)

    return Model([z, y], outputs, name='decoder')

class CustomSaver(Callback):
    def on_epoch_end(self, epoch, logs={}):
           if (k == (iterations-1)):
               decoder.save("outputs/models/{}/model_cbvae_6var_m{}.h5".format(folder_name,epoch))
               encoder.save("outputs/models/{}/encoder_cbvae_6var_m{}.h5".format(folder_name,epoch))
                
decoderSaver = CustomSaver()
iterations = 3
history = History()
k=0

class LossLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            epoch_loss = logs.get('loss')
            print(f"Epoch {epoch+1} loss: {epoch_loss:.4f}")
            logs['epoch_loss'] = epoch_loss
            wandb.log({"loss": epoch_loss})
            
def run_cvae():
    with wandb.init(config=configs):
        #wandb.init(config=configs)
        config = wandb.config
        wandb_callback = WandbCallback(monitor='val_loss',
                                   log_weights=True,
                                   log_evaluation=True,
                                   validation_steps=5)
        epochs = config.epochs
        learnrate = config.learning_rate

        encoder = create_encoder(Input(shape=(input_dim,)), Input(shape=(1,)), config.latent_dim)
        decoder = create_decoder(Input(shape=(config.latent_dim,)), Input(shape=(1,)))

        # cvae.compile(optimizer=opt, loss=vae_loss)
        X_input = Input(shape=(input_dim,))
        y_input = Input(shape=(1,))
        z_mean, z_log_var, z = encoder([X_input,y_input])
        outputs = decoder([z,y_input])
        cvae = Model([X_input, y_input], outputs)


        def mse_loss_fn(x,  x_decoded_mean):
            mse_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.mse(x, x_decoded_mean)))
            return mse_loss
    
        def kl_loss_fn(x,  x_decoded_mean):
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            return kl_loss

        def vae_loss(x, x_decoded_mean):
            mse_loss = mse_loss_fn(x, x_decoded_mean)
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            loss = K.mean((1-config.beta)*mse_loss + config.beta*kl_loss)
            return loss

        opt = Adam(learning_rate=learnrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        checkpointer = ModelCheckpoint(filepath='outputs/models/%s/cbvae_LHCO2020_20d_e-6.hdf5'%(folder_name), verbose=1, save_best_only=True)
        opt = Adam(learning_rate=learnrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        
        # tf.keras.backend.clear_session()
        cvae.compile(loss=vae_loss, optimizer=opt, metrics=[mse_loss_fn, kl_loss_fn])
        cvae.fit([x_train, y_train], x_train,
                epochs=epochs,
                batch_size=config.batch_size,
                validation_data=([x_test, y_test], x_test),
                callbacks = [checkpointer, history, decoderSaver, wandb_callback, LossLogger()])
        cvae.load_weights('outputs/models/%s/cbvae_LHCO2020_20d_e-6.hdf5'%(folder_name))

        
wandb.agent(sweep_id, function=run_cvae, count=1)
