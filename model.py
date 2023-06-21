import tensorflow as tf
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import sys

rescale = layers.Rescaling(scale=1./127.5, offset=-1)


class Model(keras.Model):
   def __call__(self, x, training):
      return self.base_model(x, training)
   def __init__(self):
      super(Model, self).__init__()
      self.noise_steps = 50
      self.optimizer = tf.keras.optimizers.Adam(1e-4)

      self.base_model = keras.Sequential([
         #inputs a noisy 64x64x3 image
         layers.Conv2D(64, (3,3), activation='relu', padding='same'),
         layers.MaxPooling2D((2,2), padding='same'),

         layers.Conv2D(128, (3,3), activation='relu', padding='same'),
         layers.MaxPooling2D((2,2), padding='same'),

         layers.Conv2D(256, (3,3), activation='relu', padding='same'),
         layers.MaxPooling2D((2,2), padding='same'),

         layers.Conv2D(512, (3,3), activation='relu', padding='same'),
         layers.MaxPooling2D((2,2), padding='same'),

         layers.Conv2D(1024, (3,3), activation='relu', padding='same'),
         layers.MaxPooling2D((2,2), padding='same'),
         layers.Conv2D(2048, (3,3), activation='relu', padding='same'),

         layers.Conv2DTranspose(1024, 3, strides=2, activation='relu', padding='same'),
         layers.Conv2DTranspose(512, 3, strides=2, activation='relu', padding='same'),
         layers.Conv2DTranspose(256, 3, strides=2, activation='relu', padding='same'),
         layers.Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same'),
         layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same'),

         
         layers.Conv2D(3, (1,1), activation='tanh', padding='same')
     
         

         #outputs a 64x64x3 that are fed into the mu and sigma networks
      ], name="base_model")

   def summary(self):
       return self.base_model.summary()
   
   def loss(self, mu, sigma, image_batch):
      return tf.keras.losses.MeanSquaredError()
   
   @tf.function
   def train_step(self, image_batch, epoch, batch_num):
      #image_batch is -1 to 1
      loss_function = tf.keras.losses.MeanSquaredError()
      schedule = np.linspace(1., 0., self.noise_steps)
      image_noise = tf.random.uniform(image_batch.shape, minval=-1, maxval=1, dtype=tf.dtypes.float32)
      for step in range(self.noise_steps):
         noisy_images = image_batch + (image_noise * schedule[step]) #noise is between -2 and 2

         with tf.GradientTape(persistent=True) as tape:
            mu = self(noisy_images, training=True) #mu is between -1 and 1 to guess the noise being added

            loss = loss_function(mu, (image_noise * schedule[step]))
            #loss = loss_function(mu, image_noise * schedule[step])
            
            tf.print("Epoch: ",epoch, " Batch: ", batch_num, " Loss: ", loss, "                                        ", end="\r")
         base_gradient = tape.gradient(loss, self.base_model.trainable_variables)
         self.optimizer.apply_gradients(zip(base_gradient, self.base_model.trainable_variables))
      tf.print("\n", end="\r")

   def inference(self, image_batch):
      noise = image_batch +  tf.random.uniform(image_batch.shape, minval=-1, maxval=1, dtype=tf.dtypes.float32)
      noise = layers.Rescaling(scale=0.5)(noise)
      rescale_noise = layers.Rescaling(scale=127.5, offset=1.0)
      for step in range(self.noise_steps):
         
         image_noise = rescale_noise(noise/2).numpy().astype(np.uint8)
         plt.imshow(image_noise[0])
         plt.show(block=False)
         plt.savefig("./Data/Results/{}.jpg".format(step))

         mu = self(noise, training=False)
        
         noise = tf.clip_by_value(noise - mu, -2, 2)
         mu = rescale_noise(mu).numpy().astype(np.uint8)
         print("Max noise: ", np.max(mu), " Min: ", np.min(mu))

