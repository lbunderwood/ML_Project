# Created 2022-04-11 by Luke Underwood
# main.py
# script for Fundamentals of Machine Learning project

# python library inclusions
import keras as ks
import tensorflow as tf
import pandas as pd
import numpy as np


# ---------Important functions----------

# import_dataset function
# returns array containing full dataset
def import_dataset():
    return "data", "predictions"


# preprocessing function
# takes array of full dataset as input
# returns array with preprocessing completed
def preprocessing(x):
    return x


# build_mlm function
# returns constructed mlm
def build_mlm():
    return "mlm"


# I think this is a one-liner anyway, might not need a function
def train(mlm, x, y):
    return mlm


# output_results function
# outputs predictions to csv, generates graphics and console output
def output_results(mlm, x, y):
    print("Results: ", mlm, x, y)


# run the program if this is the main script
if __name__ == '__main__':
    train_x, train_y = import_dataset()
    train_x = preprocessing(train_x)
    mlm = build_mlm()
    mlm = train(mlm, train_x, train_y)
    output_results(mlm, train_x, train_y)
