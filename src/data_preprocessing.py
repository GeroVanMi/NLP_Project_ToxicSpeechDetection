import csv
import time
from os import PathLike

import nltk
import numpy as np
import fasttext
from fasttext.FastText import _FastText

from helpers import limit_documents

columns = ["id", "comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
COMMENT_TEXT_INDEX = 1
TOKENS_INDEX = 8


def read_file(file_path: str | PathLike[str]) -> []:
    documents = []
    with open(file_path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for (index, row) in enumerate(reader):
            if index != 0:
                documents.append(row)

    return documents


def extract_tokens(documents: []) -> []:
    """
    :param documents:
    :return:
    """
    start = time.time()

    counter = 0
    errors = 0
    percentage = 0

    for document in documents:
        counter += 1

        if counter % round(len(documents) / 100) == 0:
            percentage += 1
            counter = 0
            print(f"Extracting tokens: {percentage}%")

        try:
            document.append(nltk.tokenize.regexp_tokenize(document[COMMENT_TEXT_INDEX], r'[a-zA-Z]+'))
        except TypeError as e:
            errors += 1
            print(e)
            print(document[COMMENT_TEXT_INDEX])
            print(type(document[COMMENT_TEXT_INDEX]))
    end = time.time()
    total_time = end - start
    print(f"Extracting tokens finished in ~{round(total_time, 4)}s")
    print(f"Encountered {errors} errors.\n")
    return documents


def process_tokens(documents: []) -> []:
    print("Processing tokens")
    for document in documents:
        tokens = document[TOKENS_INDEX]
        document[TOKENS_INDEX] = [token.lower() for token in tokens]
    return documents


def generate_bag_of_tokens(documents: []) -> {}:
    print("Generating bag of tokens.")
    bag_of_tokens = {}
    counter = 0
    for document in documents:
        tokens = document[TOKENS_INDEX]
        for token in tokens:
            if token not in bag_of_tokens:
                bag_of_tokens[token] = counter
                counter += 1

    return bag_of_tokens


def vectorize_documents(documents: [], vectorize_model: _FastText) -> []:
    """
    Converts the tokens into a number that can be used for training a neural network.

    :param documents:
    :param vectorize_model:
    :return:
    """
    print("Vectorizing documents")

    for document in documents:
        tokens = document[TOKENS_INDEX]
        token_vectors = np.ndarray(shape=(len(tokens), 100))

        for (index, token) in enumerate(tokens):
            token_vectors[index] = vectorize_model.get_word_vector(token)

        document.append(token_vectors)

    # Old One-Hot encoding implementation
    # for document in documents:
    #     token_vector = np.zeros(len(bag_of_tokens), dtype=int)
    #     tokens = document[TOKENS_INDEX]
    #
    #     for token in tokens:
    #         if token in bag_of_tokens:
    #             token_vector[bag_of_tokens[token]] = 1
    #         else:
    #             print(f"Vectorizing documents: Should remove token: {token}")
    #
    #     document.append(token_vector)
    return documents


def save_documents(documents: [], file_path: str | PathLike[str]):
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

        percentage = 0
        counter = 0
        for document in documents:
            counter += 1
            if counter % round(len(documents) / 100) == 0:
                percentage += 1
                counter = 0
                print(f"Saving documents: {percentage}%")

            file.write("\n")
            save_fields = [
                document[0],
                document[2],
                document[3],
                document[4],
                document[5],
                document[6],
                np.array2string(document[-1], max_line_width=np.inf, separator=" ")
            ]

            file.write(";".join(save_fields))
    end = time.time()
    total_time = end - start
    print(f"Saving documents finished in ~{round(total_time, 4)}s")


def save_bag_of_tokens(bag_of_tokens: dict, file_path: str | PathLike[str]):
    with open(file_path, mode='w') as file:
        for word in bag_of_tokens.keys():
            file.write(word + " ")


def train_fast_text_model(bag_of_tokens_file_path: str | PathLike[str],
                          model_path: str | PathLike[str] = '../data/fasttext/model.bin'):
    model = fasttext.train_unsupervised(bag_of_tokens_file_path, model='skipgram', minCount=1)
    model.save_model(model_path)
    return model


def main():
    # TODO: We want to measure the time and write a log of the runs so that these can be used for visualizations
    file_path = '../data/kaggle/train.csv'
    save_file_path = '../data/processed/train.csv'
    bag_of_words_file_path = '../data/fasttext/bag_of_words.txt'
    documents = read_file(file_path)

    # Only for testing
    documents = limit_documents(documents, 1000)

    documents = extract_tokens(documents)
    documents = process_tokens(documents)

    # Bag of Tokens related
    bag_of_tokens = generate_bag_of_tokens(documents)
    save_bag_of_tokens(bag_of_tokens, bag_of_words_file_path)
    # TODO: Handle the error properly, when the bag of words path doesn't exist
    vectorize_model = train_fast_text_model(bag_of_words_file_path)

    documents = vectorize_documents(documents, vectorize_model)
    save_documents(documents, save_file_path)


if __name__ == '__main__':
    main()
