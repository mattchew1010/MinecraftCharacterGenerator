import tensorflow as tf
from model import Model

noise_dim = 128
batch_size = 32
ds = tf.keras.utils.image_dataset_from_directory("./Data/Avatars", image_size=(64,64), batch_size=batch_size, labels=None).take(3000).cache().prefetch(tf.data.AUTOTUNE)

Model = Model(noise_dim)

Model.discriminator(Model(tf.random.normal([batch_size, noise_dim])))
print(Model.generator.model.summary())
print(Model.discriminator.model.summary())

Model.fit(ds, epochs=1000, batch_size=batch_size)