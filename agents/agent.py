import tensorflow as tf
import numpy as np
import datetime

class Agent:

    epochs = 100
    steps_per_epoch = 100
    logging = False
    log_dir="logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = []

    def __init__(self, model):
        self.model = model
        self.datasetX, self.datasetY = self.create_dataset(100000, 200000)
        self.testsetX, self.testsetY = self.create_dataset(30500, 31000)

        if self.logging:
            self.callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1))

    def train(self):
        model_history = self.model.model.fit(
            self.datasetX,
            self.datasetY,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=(self.testsetX, self.testsetY),
            callbacks=self.callbacks
        )

    def create_dataset(self, start, end):
        datasetX = []
        datasetY = []
        for i in range(start, end):
            datasetX.append(self.model.bin_encode(i))
            datasetY.append(self.fizzbuzz_labeler(i))
        
        return np.array(datasetX), np.array(datasetY)
    
    def fizzbuzz_labeler(self, n):
        if (n%3 == 0 and n%5 == 0):
            return [0, 0, 0, 1] #'fizzbuzz'
        elif (n%3 == 0):
            return [0, 0, 1, 0] #'fizz'
        elif (n%5 == 0):
            return [0, 1, 0, 0] #'buzz'
        else:
            return [1, 0, 0, 0]

