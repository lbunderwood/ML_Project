# Created 2022-04-11 by Luke Underwood
# main.py
# script for Fundamentals of Machine Learning project

# python library inclusions
import keras as ks
import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn as skl
from sklearn import decomposition

# ---------Global variables-------------
FEATURE_COUNT = 4096 + 512


# ---------Important functions----------

# import_dataset function
# returns arrays containing test data and predictions
def import_dataset():
    print("\nStarted data import...")
    data = pd.read_csv("training1.csv")
    x = data.iloc[:, :FEATURE_COUNT]
    y = data.iloc[:, FEATURE_COUNT:]
    print("Data import complete!\n")

    return x.values, y.values


# preprocessing function
# takes array of full dataset as input
# returns array with preprocessing completed
def preprocessing(x, y, cnn_features=4096, gist_features=512):
    print("\nStarted preprocessing...")
    # throw out confidence labels for now
    y = y[:, 0]

    # normalize features to mean 0, sd 1
    features = x.shape[1]
    for i in range(features):
        feature = x[:, i]
        x[:, i] = (feature - np.mean(feature)) / np.std(feature)

    # perform PCA for dimensionality reduction, keeping CNN and GIST features separate
    cnn_count = 4096
    x_cnn = x[:, :cnn_count]
    x_gist = x[:, cnn_count:]

    pca_cnn = skl.decomposition.PCA(n_components=cnn_features)
    pca_cnn.fit(x_cnn)
    x_cnn = pca_cnn.transform(x_cnn)

    pca_gist = skl.decomposition.PCA(n_components=gist_features)
    pca_gist.fit(x_gist)
    x_gist = pca_gist.transform(x_gist)

    x = np.append(x_cnn, x_gist, axis=1)
    print("x size = ", x.shape)

    # shuffle data
    x, y = skl.utils.shuffle(x, y)

    def divide(arr):
        length = arr.shape[0]
        return [arr[:length // 3],
                arr[length // 3:2 * length // 3],
                arr[2 * length // 3:]]

    print("Preprocessing complete!\n")
    return divide(x), divide(y)


# build_mlm function
# returns constructed mlm
def build_mlm(hidden_layers=2, layer_size=100, input_size=FEATURE_COUNT):
    print("\nStarted MLM construction...")
    # layers -  input layer: has shape matching number of features
    #           hidden layers: have number and shape specified in arguments, and relu activation
    #           output layer: single neuron, sigmoid activation
    layers = [tf.keras.Input(shape=(input_size,))]
    for layer_num in range(0, hidden_layers):
        layers.append(tf.keras.layers.Dense(layer_size, activation="relu"))
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))

    model = tf.keras.Sequential(layers)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("MLM construction complete!\n")
    return model


# train function
# takes mlm, data, and true labels as input
# returns trained mlm
def train(model, x, y):
    print("\nStarted training...")
    model.fit(x, y, batch_size=FEATURE_COUNT//8, validation_split=0.2, epochs=10, shuffle=True)
    print("Training complete!\n")
    return model


# output_results function
# outputs predictions to csv, generates graphics and console output
def output_results(model, x, y):
    print("\nStarted result output...")
    print("Results: ", model.evaluate(x, y))
    print("Result output complete!\n")


# run the program if this is the main script
if __name__ == '__main__':
    full_x, full_y = import_dataset()

    cnn_feature_nums = [1, 2, 3, 4, 6, 8, 10]
    gist_feature_nums = [1, 2, 3, 4, 6, 8, 10]
    results = np.zeros((7, 7, 2))
    i, j = 0, 0
    for cnn_features in cnn_feature_nums:

        for gist_features in gist_feature_nums:
            x_sets, y_sets = preprocessing(full_x, full_y, cnn_features, gist_features)
            result_sets = np.array([])
            # build 3 mlms, train each on 2/3 sets, save third for testing
            set_count = 3
            for k in range(set_count):
                print("\nStarting dataset number ", k, " with cnn features = ", cnn_features, " gist features = ", gist_features)

                x_train = np.concatenate((x_sets[k], x_sets[(k+1) % set_count]))
                y_train = np.concatenate((y_sets[k], y_sets[(k+1) % set_count]))
                x_test = x_sets[(k+2) % set_count]
                y_test = y_sets[(k+2) % set_count]

                mlm = build_mlm(input_size=x_test.shape[1])
                mlm = train(mlm, x_train, y_train)
                output_results(mlm, x_test, y_test)
                result_sets = np.append(result_sets, mlm.evaluate(x_test, y_test))

            result_sets = result_sets.reshape((3, 2))
            results[i, j, 0] = np.average(result_sets[:, 0])
            results[i, j, 1] = np.average(result_sets[:, 1])
            j += 1
        j = 0
        i += 1
    print(results)
