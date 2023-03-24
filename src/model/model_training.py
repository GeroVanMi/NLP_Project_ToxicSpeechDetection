import math
import random

import numpy as np
from alive_progress import alive_bar
from keras import Sequential, Input, Model
from keras.layers import Dense
from sklearn.metrics import log_loss, accuracy_score, f1_score

from Document import load_documents, limit_documents, extract_training_data, Document
from Log import Log
from bag_of_tokens import read_bag_of_tokens
from model.model_evaluation import reshape_is_toxic_vector


def create_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(1000, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    return model


def test_model(model: Model, test_documents: list[Document], bag_of_tokens: dict, desired_batch_size=500):
    max_size = 2500
    random.shuffle(test_documents)
    test_documents = test_documents[0:max_size]

    number_of_batches = len(test_documents) / desired_batch_size
    losses = []
    accuracies = []
    f_scores = []

    with alive_bar(math.ceil(number_of_batches), title='Testing model') as update_bar:
        for batch_number in range(math.ceil(number_of_batches)):
            update_bar()
            start = batch_number * desired_batch_size
            stop = start + desired_batch_size

            document_batch = test_documents[start:stop]
            if len(document_batch) < 2:
                continue

            x_test, y_test = extract_training_data(document_batch, bag_of_tokens)
            prediction = model.predict_on_batch(x_test)

            y_true = reshape_is_toxic_vector(y_test)
            y_pred = reshape_is_toxic_vector(prediction)

            losses.append(log_loss(y_test, prediction))
            accuracies.append(accuracy_score(y_true, y_pred))
            f_scores.append(f1_score(y_true, y_pred))

    return {
        'loss': np.array(losses).mean(),
        'accuracy': np.array(accuracies).mean(),
        'f1_score': np.array(f_scores).mean(),
    }


def train_model(
        logger: Log,
        desired_batch_size=500,
        number_of_epochs=5,
        limit: int = None,
):
    training_data_path = f'{logger.log_path}/data/train.csv'
    test_data_path = f'{logger.log_path}/data/test.csv'

    bag_of_tokens_file_path = f'{logger.log_path}/data/bag_of_words.json'
    train_documents = load_documents(training_data_path)
    test_documents = load_documents(test_data_path)

    if limit is not None:
        train_documents = limit_documents(train_documents, limit)

    bag_of_tokens = read_bag_of_tokens(bag_of_tokens_file_path)

    model = create_model(len(bag_of_tokens))
    logger.log_model_structure(model)

    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

    number_of_batches = len(train_documents) / desired_batch_size

    for batch_number in range(math.ceil(number_of_batches)):
        start = batch_number * desired_batch_size
        stop = start + desired_batch_size

        document_batch = train_documents[start:stop]

        if len(document_batch) < 2:
            continue

        total_amount_of_samples = start + len(document_batch)
        print(f'{total_amount_of_samples}/{len(train_documents)}')

        x_train, y_train = extract_training_data(document_batch, bag_of_tokens)

        history = model.fit(
            x_train,
            y_train,
            verbose=1,
            epochs=number_of_epochs,
            validation_split=.1,
        )

        test_scores = test_model(model, test_documents, bag_of_tokens)

        training_scores = {
            'loss': np.array(history.history['loss']).mean(),
            'accuracy': np.array(history.history['accuracy']).mean(),
            'validation_loss': np.array(history.history['val_loss']).mean(),
            'validation_accuracy': np.array(history.history['val_accuracy']).mean(),
        }
        logger.log_model_scores(total_amount_of_samples, training_scores, test_scores)

    logger.save_model(model)
