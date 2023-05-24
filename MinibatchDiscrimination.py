import tensorflow as tf

class MinibatchDiscrimination(tf.keras.layers.Layer):
    def __init__(self, num_kernels, kernel_dim, **kwargs):
        super(MinibatchDiscrimination, self).__init__(**kwargs)
        self.num_kernels = num_kernels
        self.kernel_dim = kernel_dim
    
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[-1], self.num_kernels, self.kernel_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
    
    def call(self, inputs):
        activation = tf.tensordot(inputs, self.kernel, [[1], [0]])
        diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, perm=[1,2,0]), 0)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), axis=2)
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), axis=2)
        return tf.concat([inputs, minibatch_features], axis=1)