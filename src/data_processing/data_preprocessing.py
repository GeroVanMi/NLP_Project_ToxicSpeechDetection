import csv
import time
from os import PathLike

import numpy as np
from alive_progress import alive_bar
from sklearn.model_selection import train_test_split

from Document import Document, limit_documents
from Log import Log
from Settings import Settings
from bag_of_tokens import generate_bag_of_tokens, save_bag_of_tokens
from data_processing.document_vectorization import vectorize_documents
from data_processing.documents_processing import process_documents
from data_processing.oversampling import oversample

columns = ["id", "comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def load_documents(file_path, logger=None):
    read_start_time = time.time()
    documents = []
    with open(file_path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for (index, row) in enumerate(reader):
            if index != 0:
                documents.append(Document(
                    row[0],
                    row[1],
                    int(row[2]),
                    int(row[3]),
                    int(row[4]),
                    int(row[5]),
                    int(row[6]),
                    int(row[7]),
                ))

    if logger is not None:
        logger.log_data_processing(f"Reading files:{time.time() - read_start_time}")
    return documents


def save_documents(documents, file_path, logger=None):
    start = time.time()

    with open(file_path, mode='w') as file:
        file.write(";".join([
            "id",
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
            "tokens_vector"]
        ))
        np.set_printoptions(threshold=np.inf)

        empty_docs = 0
        with alive_bar(len(documents), title="Saving documents") as update_bar:
            for document in documents:
                update_bar()

                if len(document.token_vector) != 0:
                    file.write("\n")
                    content = document.serialize()
                    file.write(content)
                else:
                    empty_docs += 1
    print()

    end = time.time()
    total_time = end - start

    if logger is not None:
        logger.log_data_processing(f"Saving documents:{total_time}")
        logger.log_data_processing(f"Empty documents:{empty_docs}")


# def train_fast_text_model(
#         bag_of_tokens_file_path: str | PathLike[str],
#         model_path: str | PathLike[str] = '../data/fasttext/model.bin'
# ):
#     model = fasttext.train_unsupervised(bag_of_tokens_file_path, model='skipgram', minCount=1)
#     model.save_model(model_path)
#     return model


def process_data(root_path, logger, settings, limit=None):
    print()
    print("Started data processing.\n")

    file_path = root_path + '/data/kaggle/train.csv'

    bag_of_words_output_path = logger.log_path + 'data/bag_of_words.json'
    train_output_path = logger.log_path + 'data/train.csv'
    test_output_path = logger.log_path + 'data/test.csv'

    start_time = time.time()

    documents = load_documents(file_path)

    if limit is not None:
        documents = limit_documents(documents, limit)

    documents = process_documents(documents, logger, settings)

    # Bag of Tokens related
    bag_of_tokens = generate_bag_of_tokens(documents)
    save_bag_of_tokens(bag_of_tokens, bag_of_words_output_path)

    # TODO: Handle the error properly, when the bag of words path doesn't exist
    # vectorize_model = train_fast_text_model(bag_of_words_file_path)

    documents = vectorize_documents(documents, bag_of_tokens)

    train_documents, test_documents = train_test_split(documents, test_size=0.2)
    if settings.oversample:
        train_documents = oversample(train_documents)

    save_documents(train_documents, train_output_path, logger)
    save_documents(test_documents, test_output_path, logger)

    # Save the total execution time to the log
    total_time = time.time() - start_time

    if logger is not None:
        logger.log_data_processing(f"Total time:{total_time}")
