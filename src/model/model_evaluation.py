from os import PathLike

import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, f1_score

from BagOfTokens import read_bag_of_tokens
from Document import load_documents, limit_documents, extract_training_data
from Log import Log


def plot_test_loss(test_loss, save_as_file=True) -> None:
    plt.plot(test_loss, label="Test loss")
    plt.xlabel("Batch Number")
    plt.ylabel("Cross-Entropy Loss")
    if save_as_file:
        plt.savefig('test_loss.png')
    else:
        plt.show()


def reshape_is_toxic_vector(x):
    y = np.ndarray((x.shape[0], 1))
    for i in range(len(x)):
        entry = x[i]
        is_not_toxic_probability = entry[0]
        is_toxic_probability = entry[1]

        is_toxic = 0
        if is_toxic_probability > is_not_toxic_probability:
            is_toxic = 1

        y[i] = is_toxic

    return y


def evaluate_model(logger: Log, limit: int = None):
    model = keras.models.load_model(f"{logger.log_path}toxic_detection_model")
    model.summary()

    weights, biases = model.layers[0].get_weights()
    logger.log_model_evaluation(f"Weights mean:{weights.mean()}")
    logger.log_model_evaluation(f"Weights standard deviation:{weights.std()}")

    file_path = logger.log_path + 'data/train.csv'
    bag_of_tokens_file_path = logger.log_path + 'data/bag_of_words.json'
    documents = load_documents(file_path)

    if limit is not None:
        documents = limit_documents(documents, limit)

    bag_of_tokens = read_bag_of_tokens(bag_of_tokens_file_path)

    x_test, y_test = extract_training_data(documents, bag_of_tokens)
    # TODO: It would be nice to see how the test score evolves alongside the model during training?
    # TODO: This also needs to be split into separate chunks => + Call them chunks instead of batches?
    # TODO: Also log the predictions, to see if they're correct?
    prediction = model.predict(x_test)

    y_true = reshape_is_toxic_vector(y_test)
    y_pred = reshape_is_toxic_vector(prediction)

    test_loss = log_loss(y_test, prediction)
    accuracy = accuracy_score(y_true, y_pred)
    f_score = f1_score(y_true, y_pred)

    logger.log_model_evaluation(f"Test-Loss:{test_loss}")
    logger.log_model_evaluation(f"Test-Accuracy:{accuracy}")
    logger.log_model_evaluation(f"Test-F1-Score:{f_score}")
