import tensorflow as tf
from model import Model
import matplotlib.pyplot as plt
import time
import numpy as np
batch_size = 16

Model = Model()
results = Model(tf.random.normal((batch_size, 64, 64, 3)), training=False).numpy()
print(np.max(results), np.min(results))
print(Model.summary())
checkpoint = tf.train.Checkpoint(base=Model.base_model, optimizer=Model.optimizer
                                 )
checkpoint.restore(tf.train.latest_checkpoint('./Data/Checkpoints'))

Model.inference(tf.zeros((batch_size, 64, 64, 3)))
