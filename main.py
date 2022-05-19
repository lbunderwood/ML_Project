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


# normalize function
# normalizes both arrays based on mean, sd of the first one
# returns both arrays
def normalize(x, x2):
    print("Normalizing data...")
    # normalize features to mean 0, sd 1
    # manipulate x2 by the same amount, since they should have similar mean, sd
    # and we want estimates to make sense/apply evenly during KNN
    for i in range(FEATURE_COUNT):
        feature = x[:, i]
        x[:, i] = (feature - np.mean(feature)) / np.std(feature)
        x2[:, i] = (x2[:, i] - np.mean(feature)) / np.std(feature)
    return x, x2


# k nearest neighbors function
# performs KNN imputing on second array based on data in first array
# returns second array with imputing done
def k_nearest_neighbors(x, x2, k_neighbors=5):
    print("Performing KNN imputing...")
    # do KNN imputing on x2
    for row2 in x2:
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
            dist = np.linalg.norm(row2_nums - row_nums)
            distances[i] = dist

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
    return x2


# dimensionality reduction function
# takes two arrays, PCA is fit to first the array, used on the second array
# returns second array after dimensionality reduction is performed
def dimensionality_reduction(x, x2, cnn_features=4, gist_features=4):
    print("Performing PCA dimensionality reduction...")
    # perform PCA for dimensionality reduction, keeping CNN and GIST features separate
    cnn_count = 4096
    x_cnn = x[:, :cnn_count]
    x_gist = x[:, cnn_count:]
    x2_cnn = x2[:, :cnn_count]
    x2_gist = x2[:, cnn_count:]

    pca_cnn = skl.decomposition.PCA(n_components=cnn_features)
    pca_cnn.fit(x_cnn)
    x2_cnn = pca_cnn.transform(x2_cnn)

    pca_gist = skl.decomposition.PCA(n_components=gist_features)
    pca_gist.fit(x_gist)
    x2_gist = pca_gist.transform(x2_gist)

    x2 = np.append(x2_cnn, x2_gist, axis=1)
    return x2


# preprocessing function for training data
# takes arrays with complete and incomplete data sets with separate labels
# returns arrays with preprocessing completed
def preprocessing_train(x, y, x2, y2, features=3):
    print("\nStarted preprocessing...")
    # throw out confidence labels
    y = y[:, 0]
    y2 = y2[:, 0]

    x, x2 = normalize(x, x2)
    x2 = k_nearest_neighbors(x, x2)

    # combine x, x2 and y, y2
    x = np.append(x, x2, axis=0)
    y = np.append(y, y2)

    x = dimensionality_reduction(x, x, cnn_features=features, gist_features=features)

    # shuffle data
    x, y = skl.utils.shuffle(x, y)

    # split data into three sets
    length = x.shape[0]
    split = [length//3, 2*length//3]
    x_split = np.split(x, split)
    y_split = np.split(y, split)
    print("Preprocessing complete!\n")
    return x_split, y_split


# preprocessing function for test data
# takes the 600-length dataset and the test dataset as arguments
# returns a preprocessed test dataset
def preprocessing_test(x, x2):
    print("\nStarted preprocessing...")
    x, x2 = normalize(x, x2)
    x2 = k_nearest_neighbors(x, x2)
    x2 = dimensionality_reduction(x, x2)
    print("Preprocessing complete!\n")
    return x2


# build_mlm function
# returns constructed mlm
def build_mlm(hidden_layers=2, layer_size=50, input_size=FEATURE_COUNT):
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
    model.fit(x, y, validation_split=0.2, epochs=10, shuffle=True)
    print("Training complete!\n")
    return model


# output_results function
# outputs predictions to csv
def output_results(models, x):
    print("\nStarted result output...")

    predictions1 = mlms[0].predict(x)
    predictions2 = mlms[1].predict(x)
    predictions3 = mlms[2].predict(x)
    predictions = np.array([predictions1, predictions2, predictions3])

    pred_vote = np.zeros_like(predictions[0])
    for i in range(len(predictions[0])):
        average = np.average(predictions[:, i])
        if average >= 0.5:
            pred_vote[i] = 1
        else:
            pred_vote[i] = 0
    np.savetxt("predictions.csv", pred_vote, delimiter=",", fmt="%d")
    print("Result output complete!\n")


# run the program if this is the main script
if __name__ == '__main__':
    full_x, full_y = import_dataset("training1.csv")
    partial_x, partial_y = import_dataset("training2.csv")
    sizes = [1, 3, 4, 10, 100]
    for size in sizes:
        x_sets, y_sets = preprocessing_train(full_x, full_y, partial_x, partial_y, features=size)

        # build 3 mlms, train each on 2/3 sets, save third for testing
        mlms = np.array([])
        set_count = 3
        for i in range(set_count):
            print("\nStarting mlp number ", i + 1)

            x_train = np.concatenate((x_sets[i], x_sets[(i+1) % set_count]))
            y_train = np.concatenate((y_sets[i], y_sets[(i+1) % set_count]))
            x_test = x_sets[(i+2) % set_count]
            y_test = y_sets[(i+2) % set_count]

            mlm = build_mlm(input_size=x_test.shape[1])
            mlm = train(mlm, x_train, y_train)
            print("Size: ", size)
            mlm.evaluate(x_test, y_test)
            mlms = np.append(mlms, mlm)

    #x_final, y_final = import_dataset("test.csv")
    #x_final = preprocessing_test(full_x, x_final)
    #output_results(mlms, x_final)




