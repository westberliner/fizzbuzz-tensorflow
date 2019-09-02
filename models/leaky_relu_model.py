import tensorflow as tf

from .base_model import BaseModel

class LeakyReluModel(BaseModel):

    model_name = 'fizzbuzz_leaky_relu'

    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_shape=(self.num_digits,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dense(4, activation='softmax')
        ])
        BaseModel.__init__(self)
