import tensorflow as tf
from model import Model
import matplotlib.pyplot as plt

batch_size = 32
ds = tf.keras.utils.image_dataset_from_directory("./Data/Avatars", image_size=(64,64), batch_size=batch_size, labels=None).take(30).cache().prefetch(tf.data.AUTOTUNE)

Model = Model()
#print(Model(tf.random.normal((batch_size, 64, 64, 3))))
#print(Model.summary())
checkpoint = tf.train.Checkpoint(base=Model.base_model, mu=Model.mu_model, sigma=Model.sigma_model, optimizer=Model.optimizer
                                 )
checkpoint.restore(tf.train.latest_checkpoint('./Data/Checkpoints'))
for epoch in range(1000):
   for batch_num, image_batch in enumerate(ds):
      print("Final loss: ", Model.train_step(image_batch))
   Model.inference(image_batch) 
   checkpoint.save('./Data/Checkpoints/ckpt')
