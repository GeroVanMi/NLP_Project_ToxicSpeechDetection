import json

from alive_progress import alive_bar

from Document import Document
from Log import Log


def read_bag_of_tokens(file_path):
    with open(file_path, mode='r') as file:
        return json.load(file)


def generate_bag_of_tokens(documents, logger=None):
    """

    :param Log logger:
    :param list[Document] documents:
    :return:
    """
    bag_of_tokens = {}
    # Words are numbered from one upward, 0 is reserved for non-existing tokens.
    counter = 1
    with alive_bar(len(documents), title="Generating bag of tokens:") as update_bar:
        for document in documents:
            update_bar()
            tokens = document.tokens

            for token in tokens:
                if token not in bag_of_tokens:
                    bag_of_tokens[token] = counter
                    counter += 1
    print()

    if logger is not None:
        logger.log_data_processing(f"Vocabulary:{len(bag_of_tokens)}")

    return bag_of_tokens


def save_bag_of_tokens(bag_of_tokens, file_path):
    """
    :param bag_of_tokens:
    :param file_path:
    :return:
    """
    with open(file_path, mode='w') as file:
        json.dump(bag_of_tokens, file)
