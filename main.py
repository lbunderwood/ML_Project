# Created 2022-04-11 by Luke Underwood
# main.py
# script for Fundamentals of Machine Learning project

# python library inclusions
import keras as ks
import tensorflow as tf
import pandas as pd
import numpy as np

# ---------Global variables-------------
FEATURE_COUNT = 4096 + 512


# ---------Important functions----------

# import_dataset function
# returns arrays containing test data and predictions
def import_dataset():
    data = pd.read_csv("training1.csv")
    x = data.iloc[:, :FEATURE_COUNT]
    y = data.iloc[:, FEATURE_COUNT:]

    return x.values, y.values


# preprocessing function
# takes array of full dataset as input
# returns array with preprocessing completed
def preprocessing(x):
    return x


# build_mlm function
# returns constructed mlm
def build_mlm():
    # layers - one for input, two hidden layers of 100
    layers = [tf.keras.Input(shape=(FEATURE_COUNT,)),
              tf.keras.layers.Dense(100, activation="relu"),
              tf.keras.layers.Dense(100, activation="relu"),
              tf.keras.layers.Dense(1, activation="heaviside")
              ]
    model = tf.keras.Sequential(layers)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# train function
# takes mlm, data, and true labels as input
# returns trained mlm
def train(model, x, y):
    model.fit(x, y, batch_size=FEATURE_COUNT//8, epochs=10, shuffle=True)
    return model


# output_results function
# outputs predictions to csv, generates graphics and console output
def output_results(model, x, y):
    print("Results: ", x.shape, y.shape,
          x[0, 0], x[0, -1], x[-1, 0], x[-1, -1],
          y[0, 0], y[0, 1], y[-1, 0], y[-1, 1])


# run the program if this is the main script
if __name__ == '__main__':
    train_x, train_y = import_dataset()
    train_x = preprocessing(train_x)
    mlm = build_mlm()
    mlm = train(mlm, train_x, train_y)
    output_results(mlm, train_x, train_y)
