import numpy as np
import tensorflow as tf
import glob
main_path = 'data'
folders = ['non_smile', 'smile']
img_height, img_width = 128, 128
# the batch size used for training
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
  main_path + '/',
  seed=123,
  validation_split=0.2,
  subset='training',
  image_size=(img_height, img_width),
  batch_size=batch_size)

# create a validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
  main_path + '/',
  subset='validation',
  validation_split=0.2,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Scaling the images for better performance.
normalization_layer = tf.keras.layers.Rescaling(1./255)
def pre_process(x, y):
    x = normalization_layer(x)
    return x, y

n_train_ds = train_ds.map(lambda x, y: pre_process(x, y))
n_val_ds = val_ds.map(lambda x, y: pre_process(x, y))

input_data = tf.keras.layers.Input(shape=(img_height, img_width, 3))
encoder = tf.keras.layers.Conv2D(32, (3,3), activation='relu', strides=2)(input_data)
encoder = tf.keras.layers.BatchNormalization()(encoder)
encoder = tf.keras.layers.Dropout(0.25)(encoder)

encoder = tf.keras.layers.Conv2D(64, (3,3), activation='relu', strides=2)(encoder)
encoder = tf.keras.layers.BatchNormalization()(encoder)
encoder = tf.keras.layers.Dropout(0.25)(encoder)

encoder = tf.keras.layers.Conv2D(64, (3,3), activation='relu', strides=2)(encoder)
encoder = tf.keras.layers.BatchNormalization()(encoder)
encoder = tf.keras.layers.Dropout(0.25)(encoder)

encoder = tf.keras.layers.Conv2D(64, (3,3), activation='relu', strides=2)(encoder)
encoder = tf.keras.layers.BatchNormalization()(encoder)
encoder = tf.keras.layers.Dropout(0.25)(encoder)

encoder = tf.keras.layers.Flatten()(encoder)
encoder = tf.keras.layers.Dense(4096)(encoder)

def sample_latent_features(distribution):
    distribution_mean, distribution_variance = distribution
    batch_size = tf.keras.backend.shape(distribution_mean)
    epsilon = tf.keras.backend.random_normal(shape=batch_size, mean=0, stddev=1)
    return distribution_mean + tf.exp(0.5 * distribution_variance) * epsilon

dist_mean = tf.keras.layers.Dense(200, name='mean')(encoder)
dist_variance = tf.keras.layers.Dense(200, name='log_variance')(encoder)
latent_encoding = tf.keras.layers.Lambda(sample_latent_features)([dist_mean, dist_variance])
encoder_model = tf.keras.Model(input_data, latent_encoding)


decoder_input = tf.keras.layers.Input(shape=(200))
decoder = tf.keras.layers.Dense(4096)(decoder_input)
decoder = tf.keras.layers.Reshape((8, 8, 64))(decoder)

decoder = tf.keras.layers.Conv2DTranspose(64, (2,2), activation='relu', strides=2)(decoder)
decoder = tf.keras.layers.BatchNormalization()(decoder)
decoder = tf.keras.layers.Dropout(0.25)(decoder)

decoder = tf.keras.layers.Conv2DTranspose(64, (2,2), activation='relu', strides=2)(decoder)
decoder = tf.keras.layers.BatchNormalization()(decoder)
decoder = tf.keras.layers.Dropout(0.25)(decoder)

decoder = tf.keras.layers.Conv2DTranspose(32, (2,2), activation='relu', strides=2)(decoder)
decoder = tf.keras.layers.BatchNormalization()(decoder)
decoder = tf.keras.layers.Dropout(0.25)(decoder)

decoder_output = tf.keras.layers.Conv2DTranspose(3, (2,2), activation='relu', strides=2)(decoder)
decoder_model = tf.keras.Model(decoder_input, decoder_output)
decoder_model.summary()

optimizer = tf.keras.optimizers.legacy.Adam(lr = 0.0005)

# Reconstruction loss calculates Mean Square error (MSE) between predicted image and orginal image.
# the loss is then normalized per batch.
def get_reconstruction_loss(y_true, y_pred):
    reconstruction_loss = tf.keras.losses.mse(y_true, y_pred)
    reconstruction_loss_batch = tf.reduce_mean(reconstruction_loss)
    return reconstruction_loss_batch * img_width * img_height

# This function implements KL loss as explained in the equation above.
#
def get_kl_loss(distribution_mean, distribution_variance):
    kl_loss = 1 + distribution_variance - tf.square(distribution_mean) - tf.exp(distribution_variance)
    kl_loss_batch = tf.reduce_mean(kl_loss)
    return kl_loss_batch * (-0.5)

# Total loss is then computed as sum of reconstruction loss and KL Loss
def vae_loss(y_true, y_pred, distribution_mean, distribution_variance):
    reconstruction_loss_batch = get_reconstruction_loss(y_true, y_pred)
    kl_loss_batch = get_kl_loss(distribution_mean, distribution_variance)
    return reconstruction_loss_batch + kl_loss_batch

@tf.function
def train_step(images):
    with tf.GradientTape() as enc, tf.GradientTape() as dec:
        latent = encoder_model(images)
        generated_images = decoder_model(latent)
        loss = vae_loss(images, generated_images, latent[0], latent[1])
    gradients_of_enc = enc.gradient(loss, encoder_model.trainable_variables)
    gradients_of_dec = dec.gradient(loss, decoder_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients_of_enc, encoder_model.trainable_variables))
    optimizer.apply_gradients(zip(gradients_of_dec, decoder_model.trainable_variables))
    return loss

import tqdm

# For better clarity of images run for more epochs - 10k, 20k
# 10 K will take roughly 4 hours on V100 GPU
epochs = 20

def train(dataset, epochs):
    t_losses = []
    b_losses = []
    iteration = 0
    for epoch in tqdm.tqdm(range(epochs)):
        for image_batch, _ in dataset:
            loss = train_step(image_batch)
            b_losses.append(loss)
            iteration += 1
        if epoch % 1 == 0:
          print(f'Epoch: {epoch} | Batch Loss: {loss} | Iteration: {++iteration} | Running loss {np.average(b_losses[-1:])}')
        t_losses.append(np.average(b_losses[:1]))
    return t_losses, b_losses
t_losses, b_losses = train(n_train_ds, epochs)