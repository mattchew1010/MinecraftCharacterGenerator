import tensorflow as tf
import keras
from keras import layers
import numpy as np
from matplotlib import pyplot as plt

results = tf.random.uniform((1, 64, 64, 3), minval=-1, maxval=1, dtype=tf.dtypes.float32).numpy()

# ds = tf.keras.utils.image_dataset_from_directory("./Data/Avatars", image_size=(64,64), batch_size=16, labels=None).take(3).cache().prefetch(tf.data.AUTOTUNE)

# for image_batch in ds:
#    print(image_batch.shape)
   
#    print("Max: {} Min: {}".format(np.max(image_batch), np.min(image_batch)))
#    plt.imshow(image_batch[0].numpy().astype(np.uint8))
#    plt.show(block=False)
#    plt.pause(.5)
#    image_batch = layers.Rescaling(scale=1./127.5, offset=-1)(image_batch)
#    image_batch = image_batch + (tf.random.uniform(image_batch.shape, minval=-1, maxval=1, dtype=tf.dtypes.float32) * 0.5)
#    image_batch = layers.Rescaling(scale=0.5)(image_batch)
#    image_batch = (layers.Rescaling(scale=127.5, offset=1)(image_batch)).numpy().astype(np.uint8)
#    plt.imshow(image_batch[0])
#    plt.show(block=False)
#    plt.pause(5)
   
array = tf.TensorArray(dtype=tf.dtypes.float32, size=1, dynamic_size=True, clear_after_read=True)
array.write(0, 50)
print(array.stack().numpy())
print(array)

