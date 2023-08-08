import os
import cv2
import numpy as np
import tensorflow as tf

class Model():
    def __init__(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Rescaling(1./255, input_shape=(100, 100, 3)))
        self.model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=(100, 100, 3)))
        self.model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        self.model.add(tf.keras.layers.Dense(units=2, activation='softmax'))
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["sparse_categorical_accuracy"])
        self.model = tf.keras.models.load_model(os.getcwd())
        

    def train(self, x_train):
        self.model.fit(x_train, epochs=5)
        self.model.save(os.getcwd())

    def evaluate(self, x_test):
        return self.model.evaluate(x_test)

    def predict(self, img):
        prediccion = self.model.predict(x=np.array([np.array(cv2.resize(cv2.imread(img),(100,100)))]))
        return np.where(prediccion==prediccion.max())[1][0]


