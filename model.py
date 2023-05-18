import tensorflow as tf
import keras
from keras import layers
import shutil
import matplotlib.pyplot as plt
import os

rescale_images = tf.keras.layers.Rescaling(scale=1./255)

class Generator():
   def __call__(self, *args, **kwargs):
      return self.model(*args, **kwargs)

   def __init__(self, noise_dim):
      self.noise_dim = noise_dim
      self.model = keras.Sequential([
         #input layers
         layers.Dense(2048, input_shape=(noise_dim,), activation='relu'),
         layers.Dense(2048, activation='relu'),
         layers.BatchNormalization(),

         #final input layer with resizing
         layers.Dense(4*4*1024, activation='relu'),
         layers.BatchNormalization(),
         layers.Reshape((4,4,1024)),

         #convolutional layers
         layers.Conv2DTranspose(1024, 3, strides=(2,2), padding='same', activation='relu'),
         layers.BatchNormalization(),

         layers.Conv2DTranspose(512, 3, strides=(2,2), padding='same', activation='relu'),
         layers.BatchNormalization(),

         layers.Conv2DTranspose(128, 3, strides=(2,2), padding='same', activation='relu'),
         layers.BatchNormalization(),

         layers.Conv2DTranspose(64, 3, strides=(2,2), padding='same', activation='relu'),
         layers.BatchNormalization(),

         layers.Conv2DTranspose(3, 3, strides=(1,1), padding='same', activation='tanh'),
      ])

   def loss(self, disc_pred):
      #higher disc_pred means the discriminator thinks the image is real
      #so the loss is higher when the disc is more confident
      return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(disc_pred), disc_pred)


class Discriminator():
   def __call__(self, *args, **kwargs):
      return self.model(*args, **kwargs)
   
   def __init__(self):
      self.model = keras.Sequential([
         layers.Conv2D(64, 3, strides=(2,2), padding='same', activation='relu'),
         layers.Dropout(0.2),

         layers.Conv2D(32, 3, strides=(2,2), padding='same', activation='relu'),
         layers.Dropout(0.2),

         layers.Conv2D(16, 3, strides=(2,2), padding='same', activation='relu'),
         layers.Dropout(0.2),

         layers.Dense(128, activation='sigmoid'),
         layers.Dense(1),
      ])

   def loss(self, real_pred, gen_pred):
      #real_pred is the discriminator's prediction of the realness of the real image
      #gen_pred is the discriminator's prediction of the realness of the generated image
      real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_pred), real_pred)
      fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(gen_pred), gen_pred)
      return real_loss + fake_loss


class Model(keras.Model):
   def __call__(self, *args, **kwargs):
      return self.generator.model(*args, **kwargs)
   
   def __init__(self, noise_dim):
      super(Model, self).__init__()
      self.noise_dim = noise_dim
      self.generator = Generator(noise_dim)
      self.discriminator = Discriminator()
      self.gen_optim = tf.keras.optimizers.Adam(1e-4)
      self.disc_optim = tf.keras.optimizers.Adam(1e-5)
   @tf.function
   def train_step(self, real_images):
      noise = tf.random.normal([real_images.shape[0], self.noise_dim])
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
         gen_images = self.generator(noise, training=True)

         real_pred = self.discriminator(rescale_images(real_images), training=True)
         gen_pred = self.discriminator(gen_images, training=True)

         gen_loss = self.generator.loss(gen_pred)
         disc_loss = self.discriminator.loss(real_pred, gen_pred)
      
      gen_gradient = gen_tape.gradient(gen_loss, self.generator.model.trainable_variables)
      disc_gradient = disc_tape.gradient(disc_loss, self.discriminator.model.trainable_variables)

      self.gen_optim.apply_gradients(zip(gen_gradient, self.generator.model.trainable_variables))
      self.disc_optim.apply_gradients(zip(disc_gradient, self.discriminator.model.trainable_variables))
      return gen_loss, disc_loss

   def fit(self, dataset, epochs, batch_size):
      batch_count = dataset.cardinality().numpy()

      if os.path.exists("./Data/GeneratedImages"):
         shutil.rmtree("./Data/GeneratedImages")
      os.mkdir("./Data/GeneratedImages")

      checkpoint = tf.train.Checkpoint(generator_optimizer=self.gen_optim,
                                 discriminator_optimizer=self.disc_optim,
                                 generator=self.generator.model,
                                 discriminator=self.discriminator.model
                                 )
      checkpoint.restore(tf.train.latest_checkpoint('./Data/Checkpoints'))
      for epoch in range(epochs):
         os.mkdir("./Data/GeneratedImages/epoch_{:04d}".format(epoch))
         batch_num = 1
         for batch in dataset:
            gen_loss, disc_loss = self.train_step(batch)
            print(f"Epoch {epoch+1} | Batch: {batch_num}/{batch_count} | Generator Loss: {round(gen_loss.numpy().item(), 4)} | Discriminator Loss: {round(disc_loss.numpy().item(), 4)}", end='\r')
            batch_num += 1
         checkpoint.save('./Data/Checkpoints/ckpt')
         pred = self.generator(tf.random.normal([batch_size, self.noise_dim]), training=False)
         for i in range(pred.shape[0]):
            plt.clf()
            plt.imshow(pred[i])
            plt.axis('off')
            plt.savefig('./Data/GeneratedImages/epoch_{:04d}/{i}.png'.format(epoch, i=i), bbox_inches='tight')


if __name__ == "__main__":
    print("This file is not meant to be run directly.")