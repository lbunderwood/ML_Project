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
def import_dataset(dataset):
    print("\nStarted data import...")
    data = pd.read_csv(dataset)
    x = data.iloc[:, :FEATURE_COUNT]
    y = data.iloc[:, FEATURE_COUNT:]
    print("Data import complete!\n")

    return x.values, y.values


# preprocessing function
# takes array of full dataset as input
# returns array with preprocessing completed
def preprocessing(x, y, x2, y2, k_neighbors=5, cnn_features=4096, gist_features=512):
    print("\nStarted preprocessing...")
    # throw out confidence labels for now
    y = y[:, 0]
    y2 = y2[:, 0]

    num = 0
    # do KNN imputing on x2
    for row2 in x2:
        num += 1
        # array of distances between row2 and all rows in x
        distances = np.zeros(x.shape[0])
        # array of the values from row2 that are not nan
        row2_nums = np.zeros(FEATURE_COUNT)
        # array of indices corresponding to the values in row2_nums
        row2_idx = np.zeros_like(row2_nums)
        # counter for non-nan elements of row2
        num_count = 0
        for i in range(len(row2)):
            if not pd.isna(row2[i]):
                row2_nums[num_count] = row2[i]
                row2_idx[num_count] = i
        row2_nums = row2_nums[:num_count]
        row2_idx = row2_idx[:num_count]

        for i in range(x.shape[0]):
            # array of corresponding values to row2_nums from row
            row_nums = np.zeros_like(row2_nums)
            for j in range(len(row2_idx)):
                row_nums[j] = x[i][int(row2_idx[j])]
            # compute euclidean distance, store in distances
            dist = np.linalg.norm(row2_nums-row_nums)
            distances[i] = dist

            # check to make sure no nans make it through
            if pd.isna(dist):
                raise ValueError("NaN found in calculated distance during KNN")

        # get indices that would sort distances
        distIdx = np.argsort(distances)
        # use sorted indices to get the indices of the k nearest neighbors
        k_nearest_idx = distIdx[:k_neighbors]

        # for all nan values, calculate the average value from k nearest
        for i in range(len(row2)):
            if pd.isna(row2[i]):
                # get k nearest
                k_values = np.zeros_like(k_nearest_idx)
                for j in range(k_neighbors):
                    k_values[j] = x[k_nearest_idx[j], i]

                # average them and assign new value
                avg = np.average(k_values)
                row2[i] = avg

    # combine x, x2 and y, y2
    x = np.append(x, x2, axis=0)
    y = np.append(y, y2)

    # normalize features to mean 0, sd 1
    for i in range(FEATURE_COUNT):
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
    model.fit(x, y, batch_size=FEATURE_COUNT//8, validation_split=0.2, epochs=30, shuffle=True)
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
    full_x, full_y = import_dataset("training1.csv")
    partial_x, partial_y = import_dataset("training2.csv")

    x_sets, y_sets = preprocessing(full_x, full_y, partial_x, partial_y, cnn_features=4, gist_features=4)
    # build 3 mlms, train each on 2/3 sets, save third for testing
    set_count = 3
    for k in range(set_count):
        print("\nStarting dataset number ", k)

        x_train = np.concatenate((x_sets[k], x_sets[(k+1) % set_count]))
        y_train = np.concatenate((y_sets[k], y_sets[(k+1) % set_count]))
        x_test = x_sets[(k+2) % set_count]
        y_test = y_sets[(k+2) % set_count]

        mlm = build_mlm(input_size=x_test.shape[1])
        mlm = train(mlm, x_train, y_train)
        output_results(mlm, x_test, y_test)
