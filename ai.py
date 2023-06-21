import tensorflow as tf
from model import Model
import matplotlib.pyplot as plt
import time
from keras import layers
import numpy as np

batch_size = 32
ds = tf.keras.utils.image_dataset_from_directory("./Data/Avatars", image_size=(64,64), batch_size=batch_size, labels=None).take(100).cache().prefetch(tf.data.AUTOTUNE)


def preprocess(image_batch):
   return layers.Rescaling(scale=1./127.5, offset=-1)(image_batch)
ds = ds.map(preprocess)

Model = Model()
Model(tf.random.normal((32,64,64,3)), training=False)
print(Model.summary())
checkpoint = tf.train.Checkpoint(base=Model.base_model, optimizer=Model.optimizer
                                 )

#checkpoint.restore(tf.train.latest_checkpoint('./Data/Checkpoints'))
for epoch in range(1000):
   epoch_losses = np.array([])
   epoch_start = time.time()
   for batch_num, image_batch in enumerate(ds):
      losses = Model.train_step(image_batch, tf.constant(epoch), tf.constant(batch_num)).numpy()
      epoch_losses = np.append(epoch_losses, losses)
   plt.clf()
   plt.plot(losses)
   plt.show(block=False)
   plt.savefig(f"./Data/Plots/{epoch}.jpg")
   plt.clf()
      #print("Epoch: {} Batch: {} Time Elapsed: {} last loss: {}".format(epoch, batch_num, round(time.time() - epoch_start, 4), loss.numpy()))
   #Model.inference(image_batch) 
   checkpoint.save('./Data/Checkpoints/ckpt')
