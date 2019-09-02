import tensorflow as tf
import numpy as np
import tensorflowjs as tfjs

class BaseModel:

    model_path = 'trained_models'
    model_json_path = 'html/src/model'
    model_name = 'Base'
    num_digits = 24
    class_names = ['NAN', 'BUZZ', 'FIZZ', 'FIZZBUZZ']

    def __init__(self):
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def load(self):
        self.model = tf.keras.models.load_model('{}/{}.h5'.format(self.model_path, self.model_name))
    
    def save(self):
        self.model.save('{}/{}.h5'.format(self.model_path, self.model_name))

    def save_to_json(self):
        tfjs.converters.save_keras_model(model, self.model_json_path)

    def bin_encode(self, i):
        return [i >> d & 1 for d in range(self.num_digits)]
    
    def predict_num(self, n):
        encoded_num = np.array([self.bin_encode(n)])
        prediction = self.model.predict(encoded_num)[0]
        arg_max_prediction = tf.argmax(prediction)

        return self.class_names[arg_max_prediction]
