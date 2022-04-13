# Created 2022-04-11 by Luke Underwood
# main.py
# script for Fundamentals of Machine Learning project

# python library inclusions
import keras as ks
import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn as skl

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
def preprocessing(x, y):
    x, y = skl.utils.shuffle(x, y)

    def divide(arr):
        length = arr.shape[0]
        return [arr[:length // 3],
                arr[length // 3:2 * length // 3],
                arr[2 * length // 3:]]

    return divide(x), divide(y)

# build_mlm function
# returns constructed mlm
def build_mlm():
    # layers - one for input, two hidden layers of 100
    layers = [tf.keras.Input(shape=(FEATURE_COUNT,)),
              tf.keras.layers.Dense(100, activation="relu"),
              tf.keras.layers.Dense(100, activation="relu"),
              tf.keras.layers.Dense(1, activation="sigmoid")
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
    print("Results: ", model.evaluate(x, y))


# run the program if this is the main script
if __name__ == '__main__':
    full_x, full_y = import_dataset()
    x_sets, y_sets = preprocessing(full_x, full_y)

    # build 3 mlms, train each on 2/3 sets, save third for validation
    set_count = 3
    for i in range(set_count):
        mlm = build_mlm()

        x_train = np.concatenate((x_sets[i], x_sets[(i+1) % set_count]))
        y_train = np.concatenate((y_sets[i], y_sets[(i+1) % set_count]))
        x_valid = x_sets[(i+2) % set_count]
        y_valid = x_sets[(i+2) % set_count]

        mlm = train(mlm, x_train, y_train)
        output_results(mlm, x_valid, y_valid)
