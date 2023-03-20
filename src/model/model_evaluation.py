from os import PathLike

import keras
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

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


def evaluate_model(logger: Log, limit: int = None):
    model = keras.models.load_model(f"{logger.log_path}toxic_detection_model")
    model.summary()

    file_path = logger.log_path + 'data/train.csv'
    bag_of_tokens_file_path = logger.log_path + 'data/bag_of_words.json'
    documents = load_documents(file_path)

    if limit is not None:
        documents = limit_documents(documents, limit)

    bag_of_tokens = read_bag_of_tokens(bag_of_tokens_file_path)

    x_test, y_test = extract_training_data(documents, bag_of_tokens)
    # TODO: Maybe it would be nice to see how the test score evolves alongside the model?
    prediction = model.predict(x_test)
    test_score = (log_loss(y_test, prediction))
    print(f"SCORE:{test_score}")
