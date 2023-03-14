import csv
from os import PathLike

import numpy as np
from keras import Sequential, Input
from keras.layers import Dense

from src.helpers import print_documents
import tensorflow as tf


def read_file(file_path: str | PathLike[str]):
    documents = []
    with open(file_path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        for (index, row) in enumerate(reader):
            if index != 0:
                row[-1] = np.fromstring(row[-1].replace("[", "").replace("]", ""), dtype=int, sep=" ")
                documents.append(row)

    return documents


def create_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(5, activation="softmax"))
    return model


def extract_training_data(documents):
    x_train = []
    y_train = []

    for document in documents:
        x_train.append(document[-1])
        y_train.append([
            document[1],
            document[2],
            document[3],
            document[4],
            document[5],
        ])

    return np.array(x_train, dtype=int), np.array(y_train, dtype=int)


def main():
    file_path = '../data/processed/train.csv'
    documents = read_file(file_path)
    shape = documents[-1][-1].shape
    model = create_model(shape)
    # print_documents(documents)
    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

    x_train, y_train = extract_training_data(documents)

    history = model.fit(
        x_train,
        y_train,
        batch_size=16,
        epochs=20,
        validation_split=.1,
    )


if __name__ == '__main__':
    main()
