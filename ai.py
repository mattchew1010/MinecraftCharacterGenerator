import tensorflow as tf
from model import Model

noise_dim = 200
batch_size = 32
ds = tf.keras.utils.image_dataset_from_directory("./Data/Avatars", image_size=(64,64), batch_size=batch_size, labels=None).take(750).cache().prefetch(tf.data.AUTOTUNE)

Model = Model(noise_dim)
Model.compile()




Model.fit(ds, epochs=1000, batch_size=batch_size)