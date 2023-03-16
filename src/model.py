import csv
import json
from os import PathLike

import numpy as np
from keras import Sequential, Input
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

from Document import Document
from helpers import limit_documents

BATCH_SIZE = 500
NUMBER_OF_BATCHES = 5
EPOCHS_PER_BATCH = 5


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


def plot_learning_curves(history):
    n = len(history.history['loss'])
    plt.plot(np.arange(1, n + 1), history.history['loss'], label="training loss")
    plt.plot(np.arange(1, n + 1), history.history['val_loss'], label="validation loss")
    plt.legend()
    plt.xticks(np.arange(1, n + 1, 2))
    plt.show()


def plot_test_loss(test_loss):
    plt.plot(test_loss, label="Test loss")
    plt.xlabel("Batch Number")
    plt.ylabel("Cross-Entropy Loss")
    plt.show()


def main():
    file_path = '../data/processed/train.csv'
    bag_of_tokens_file_path = '../data/processed/bag_of_words.json'
    documents = load_documents(file_path)

    bag_of_tokens = read_bag_of_tokens(bag_of_tokens_file_path)
    # print(len(bag_of_tokens))
    # exit()
    model = create_model(len(bag_of_tokens))
    # print_documents(documents)
    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

    test_log_loss = []

    # TODO: Extract this into a separate function and use a cache, so that the model doesn't have to be
    #       retrained every single time, a change is made on the evaluation part.
    for train_index in range(NUMBER_OF_BATCHES):
        start = train_index * BATCH_SIZE
        stop = train_index * BATCH_SIZE + BATCH_SIZE

        documents_split = documents[start:stop]
        x_train, y_train = extract_training_data(documents_split, bag_of_tokens)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train)

        history = model.fit(
            x_train,
            y_train,
            batch_size=16,
            epochs=EPOCHS_PER_BATCH,
            validation_split=.1,
        )

        plot_learning_curves(history)
        prediction = model.predict(x_test)
        test_log_loss.append(log_loss(y_test, prediction))
    plot_test_loss(test_log_loss)


if __name__ == '__main__':
    main()
