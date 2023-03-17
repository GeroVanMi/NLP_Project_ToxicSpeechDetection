import csv
import json
# from fasttext.FastText import _FastText
import logging
import time
from os import PathLike

import fasttext
import numpy as np
from alive_progress import alive_bar

from Document import Document

columns = ["id", "comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def read_file(file_path: str | PathLike[str]) -> [Document]:
    read_start_time = time.time()
    documents = []
    with open(file_path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for (index, row) in enumerate(reader):
            if index != 0:
                documents.append(Document(
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    row[6],
                    row[7],
                ))

    logging.info(f"Reading files:{time.time() - read_start_time}")
    return documents


def extract_tokens(documents: [Document]) -> []:
    """
    TODO: Decide if we want to do this in the __init__ function of the Document class?

    :param documents:
    :return:
    """
    start = time.time()

    with alive_bar(len(documents), title="Extracting tokens") as update_bar:
        for document in documents:
            update_bar()
            document.tokenize()

    end = time.time()
    total_time = end - start
    logging.info(f"Extracting tokens:{total_time}")
    return documents


def process_tokens(documents: [Document]) -> [Document]:
    """
    TODO: Decide if we want to do this in the __init__ function of the Document class?
    TODO: Grouping together multiple for loops through the documents could improve performance.

    :param documents:
    :return:
    """
    with alive_bar(len(documents), title="Processing tokens") as update_bar:
        for document in documents:
            update_bar()
            document.apply_lower_case()
    return documents


def generate_bag_of_tokens(documents: [Document]) -> {}:
    """
    TODO: Add logging and documentation

    :param documents:
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

    logging.info(f"Vocabulary:{len(bag_of_tokens)}")
    return bag_of_tokens


def vectorize_documents(documents: [Document], bag_of_tokens: {}) -> []:
    """
    Converts the tokens into a number that can be used for training a neural network.
    TODO: This should be grouped with the other calls as well

    :param bag_of_tokens:
    :param documents:
    :return:
    """
    start = time.time()

    with alive_bar(len(documents), title=f'Vectorizing documents') as update_bar:
        for document in documents:
            update_bar()
            document.vectorize_tokens(bag_of_tokens)

    end = time.time()
    total_time = end - start
    logging.info(f"Vectorizing documents:{total_time}")
    return documents


def save_documents(documents: [Document], file_path: str | PathLike[str]):
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

    end = time.time()
    total_time = end - start
    logging.info(f"Saving documents:{total_time}")
    logging.info(f"Empty documents:{empty_docs}")


def save_bag_of_tokens(bag_of_tokens: dict, file_path: str | PathLike[str]):
    with open(file_path, mode='w') as file:
        json.dump(bag_of_tokens, file)
        # for word in bag_of_tokens.keys():
        #     file.write(word + " ")


def train_fast_text_model(bag_of_tokens_file_path: str | PathLike[str],
                          model_path: str | PathLike[str] = '../data/fasttext/model.bin'):
    model = fasttext.train_unsupervised(bag_of_tokens_file_path, model='skipgram', minCount=1)
    model.save_model(model_path)
    return model


def main():
    start_time = time.time()

    file_path = '../data/kaggle/train.csv'
    save_file_path = '../data/processed/train.csv'
    bag_of_words_file_path = '../data/processed/bag_of_words.json'
    documents = read_file(file_path)

    # Only for testing
    # documents = limit_documents(documents, 10000)

    documents = extract_tokens(documents)
    documents = process_tokens(documents)

    # Bag of Tokens related
    bag_of_tokens = generate_bag_of_tokens(documents)
    save_bag_of_tokens(bag_of_tokens, bag_of_words_file_path)

    # TODO: Handle the error properly, when the bag of words path doesn't exist
    # vectorize_model = train_fast_text_model(bag_of_words_file_path)

    documents = vectorize_documents(documents, bag_of_tokens)
    save_documents(documents, save_file_path)

    # Save the total execution time to the log
    total_time = time.time() - start_time
    logging.info(f"Total time:{total_time}")


if __name__ == '__main__':
    current_time = time.time()

    logging.basicConfig(
        filename=f'../logs/processing/{round(current_time)}.log',
        level=logging.INFO,
        filemode='w',
        format='%(message)s'
    )

    main()
