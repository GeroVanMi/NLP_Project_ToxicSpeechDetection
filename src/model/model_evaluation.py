from os import PathLike

import keras
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

from BagOfTokens import read_bag_of_tokens
from Document import load_documents, limit_documents, extract_training_data


def plot_test_loss(test_loss, save_as_file=True) -> None:
    plt.plot(test_loss, label="Test loss")
    plt.xlabel("Batch Number")
    plt.ylabel("Cross-Entropy Loss")
    if save_as_file:
        plt.savefig('test_loss.png')
    else:
        plt.show()


def evaluate_model(root_path: str | PathLike[str], limit: int = None, desired_batch_size=500):
    model = keras.models.load_model(f"{root_path}/data/model/toxic_detection_model")
    model.summary()

    file_path = root_path + '/data/processed/train.csv'
    bag_of_tokens_file_path = root_path + '/data/processed/bag_of_words.json'
    documents = load_documents(file_path)

    if limit is not None:
        documents = limit_documents(documents, limit)

    bag_of_tokens = read_bag_of_tokens(bag_of_tokens_file_path)
    print("EVALUATION: ####################")
    print(len(bag_of_tokens))

    test_log_loss = []
    number_of_batches = len(documents) / desired_batch_size

    x_test, y_test = extract_training_data(documents, bag_of_tokens)
    prediction = model.predict(x_test)
    test_log_loss.append(log_loss(y_test, prediction))

    # for batch_number in range(math.ceil(number_of_batches)):
    #     # On the last run, train with the remaining data points
    #
    #     start = batch_number * desired_batch_size
    #     stop = start + desired_batch_size
    #
    #     document_batch = documents[start:stop]
    #     x_test, y_test = extract_training_data(document_batch, bag_of_tokens)
    #     print(x_test.shape)
    #
    #     prediction = model.predict(x_test)
    #     test_log_loss.append(log_loss(y_test, prediction))

    # TODO: This plot naturally no longer says anything, since we are measuring this during the training phase!
    plot_test_loss(test_log_loss)


if __name__ == '__main__':
    evaluate_model('../../')
