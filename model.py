import tensorflow as tf
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import sys

rescale = layers.Rescaling(scale=1./127.5, offset=-1)


class Model(keras.Model):
   def __call__(self, x, training):
      base = self.base_model(x, training)

      mu = self.mu_model(base, training)
      sigma = self.sigma_model(base, training)

      return mu, sigma
   def __init__(self):
      super(Model, self).__init__()
      self.noise_steps = 50
      self.sigma_weight = tf.constant(1.0)
      self.optimizer = tf.keras.optimizers.Adam()

      self.base_model = keras.Sequential([
         #inputs a noisy 64x64x3 image
         
         layers.Conv2D(64, (3,3), activation='relu', padding='same'),
         layers.Conv2D(128, (3,3), activation='relu', padding='same'),
         layers.Conv2D(256, (3,3), activation='relu', padding='same'),
         layers.Conv2D(512, (3,3), activation='relu', padding='same'),
         

         #outputs a 64x64x3 that are fed into the mu and sigma networks
      ], name="base_model")
      self.sigma_model = keras.Sequential([
         #inputs a 64x64x512 image
         layers.Conv2D(128, (3,3), activation='relu', padding='same'),
         layers.Conv2D(3, (3,3), padding='same'),
         #outputs a 64x64x3 image with values between -1 and 1
      ], name="sigma_model")
      self.mu_model = keras.Sequential([
         #inputs a 64x64x512 image
         layers.Conv2D(128, (3,3), activation='relu', padding='same'),
         layers.Conv2D(3, (3,3), activation='tanh', padding='same'),
         #outputs a 64x64x3 image with values between -1 and 1
      ], name="mu_model")

   def summary(self):
       return self.base_model.summary(), self.mu_model.summary(), self.sigma_model.summary()
   
   def loss(self, mu, sigma, image_batch):
      return tf.reduce_mean(tf.square(image_batch - mu)) + tf.reduce_mean(tf.square(image_batch - sigma))
   
   @tf.function
   def train_step(self, image_batch):
      image_batch = rescale(image_batch) #rescale to -1 to 1

      schedule = np.linspace(1., 0., self.noise_steps)

      for step in range(self.noise_steps):

         noisy_images = image_batch + tf.random.normal(image_batch.shape) * schedule[step]
         with tf.GradientTape(persistent=True) as tape:

            #forward pass
            mu, sigma = self(noisy_images, training=True)
            sigma = tf.exp(sigma/2)

            mse = tf.reduce_mean(tf.square(image_batch - mu))
            mse_sigma = tf.reduce_mean(tf.square(sigma - 1)) 
            loss = mse + self.sigma_weight * mse_sigma

         base_gradient = tape.gradient(loss, self.base_model.trainable_variables)
         mu_gradient = tape.gradient(loss, self.mu_model.trainable_variables)
         sigma_gradient = tape.gradient(loss, self.sigma_model.trainable_variables)

         self.optimizer.apply_gradients(zip(base_gradient, self.base_model.trainable_variables))
         self.optimizer.apply_gradients(zip(mu_gradient, self.mu_model.trainable_variables))
         self.optimizer.apply_gradients(zip(sigma_gradient, self.sigma_model.trainable_variables))
   def inference(self, image_batch):
      noise = tf.random.normal(image_batch.shape)
      rescale_noise = layers.Rescaling(scale=127.5, offset=1)

      for step in range(self.noise_steps):
         mu, sigma = self(noise, False)
         noise =  mu + noise * sigma

         image_noise = rescale_noise(noise).numpy().astype(np.uint8)
         print("Max: ", np.max(image_noise), "Min: ", np.min(image_noise))
         plt.imshow(image_noise[0])
         plt.show(block=False)
         plt.pause(.00001)

