import csv
import json
from os import PathLike

import numpy as np
from keras import Sequential, Input
from keras.layers import Dense

from Document import Document
from helpers import limit_documents


def load_documents(file_path: str | PathLike[str]):
    documents = []
    with open(file_path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        for (index, row) in enumerate(reader):
            if index != 0:
                documents.append(Document.deserialize(row))

    return documents


def create_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(1000, activation="relu"))
    model.add(Dense(6, activation="softmax"))
    return model


def extract_training_data(documents: [Document], bag_of_tokens: {str: int}):
    x_train = []
    y_train = []

    for document in documents:
        x_train.append(document.one_hot_encode(bag_of_tokens))
        y_train.append(np.array([
            document.toxic,
            document.severe_toxic,
            document.obscene,
            document.threat,
            document.insult,
            document.identity_hate
        ]))

    return np.array(x_train, dtype=int), np.array(y_train, dtype=int)


def read_bag_of_tokens(file_path: str):
    with open(file_path, mode='r') as file:
        return json.load(file)


def main():
    file_path = '../data/processed/train.csv'
    bag_of_tokens_file_path = '../data/processed/bag_of_words.json'
    documents = load_documents(file_path)

    documents = limit_documents(documents, 1000)

    bag_of_tokens = read_bag_of_tokens(bag_of_tokens_file_path)
    # print(len(bag_of_tokens))
    # exit()
    model = create_model(len(bag_of_tokens))
    # print_documents(documents)
    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

    x_train, y_train = extract_training_data(documents, bag_of_tokens)
    print(x_train.shape)

    history = model.fit(
        x_train,
        y_train,
        batch_size=16,
        epochs=5,
        validation_split=.1,
    )


if __name__ == '__main__':
    main()
