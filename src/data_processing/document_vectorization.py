import time

from alive_progress import alive_bar

from Document import Document
from Log import Log


def vectorize_documents(documents, bag_of_tokens: {}, logger=None):
    """
    Converts the tokens into a number that can be used for training a neural network.

    :param logger:
    :param bag_of_tokens:
    :param documents:
    :return:
    """
    start = time.time()

    with alive_bar(len(documents), title=f'Vectorizing documents') as update_bar:
        for document in documents:
            update_bar()
            document.vectorize_tokens(bag_of_tokens)
    print()

    end = time.time()
    total_time = end - start
    if logger is not None:
        logger.log_data_processing(f"Vectorizing documents:{total_time}")

    return documents
