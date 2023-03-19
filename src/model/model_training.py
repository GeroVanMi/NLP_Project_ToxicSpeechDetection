import math
from os import PathLike

import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential, Input
from keras.layers import Dense

from BagOfTokens import read_bag_of_tokens
from Document import load_documents, limit_documents, extract_training_data


def create_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(1000, activation="relu"))
    model.add(Dense(6, activation="softmax"))
    return model


def plot_learning_curves(history):
    n = len(history.history['loss'])
    plt.plot(np.arange(1, n + 1), history.history['loss'], label="training loss")
    plt.plot(np.arange(1, n + 1), history.history['val_loss'], label="validation loss")
    plt.legend()
    plt.xticks(np.arange(1, n + 1, 2))
    plt.show()


def train_model(
        root_path: str | PathLike[str],
        desired_batch_size=500,
        number_of_epochs=5,
        limit: int = None,
):
    file_path = root_path + '/data/processed/train.csv'
    bag_of_tokens_file_path = root_path + '/data/processed/bag_of_words.json'
    documents = load_documents(file_path)

    if limit is not None:
        documents = limit_documents(documents, limit)

    bag_of_tokens = read_bag_of_tokens(bag_of_tokens_file_path)

    model = create_model(len(bag_of_tokens))
    print("TRAINING: ####################")
    print(len(bag_of_tokens))
    # TODO: I want this summary to be logged!
    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

    number_of_batches = len(documents) / desired_batch_size

    for batch_number in range(math.ceil(number_of_batches)):
        # On the last run, train with the remaining data points

        start = batch_number * desired_batch_size
        stop = start + desired_batch_size

        document_batch = documents[start:stop]
        x_train, y_train = extract_training_data(document_batch, bag_of_tokens)
        print(x_train.shape)

        # TODO: Find a way to merge multiple histories
        # TODO: Log / Store the training history for a given model so that stats about it can be retrieved a plotted
        # TODO: Maybe plot the history and store the file?
        _ = model.fit(
            x_train,
            y_train,
            verbose=1,
            batch_size=10,
            epochs=number_of_epochs,
            validation_split=.1,
        )

        # plot_learning_curves(history)
    model.save(root_path + '/data/model/toxic_detection_model')


if __name__ == '__main__':
    train_model('../..', 100)
