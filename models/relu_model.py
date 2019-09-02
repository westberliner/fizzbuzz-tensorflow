import tensorflow as tf

from .base_model import BaseModel

class ReluModel(BaseModel):

    model_name = 'fizzbuzz_relu'

    def __init__(self):
        self.num_digits = num_digits
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_shape=(num_digits,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(4, activation='softmax')
        ])
        BaseModel.__init__(self)
        

