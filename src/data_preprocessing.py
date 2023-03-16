import csv
import time
from os import PathLike

import fasttext
import numpy as np
from fasttext.FastText import _FastText

from Document import Document
from helpers import limit_documents

columns = ["id", "comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def read_file(file_path: str | PathLike[str]) -> [Document]:
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

    return documents


def extract_tokens(documents: [Document]) -> []:
    """
    TODO: Decide if we want to do this in the __init__ function of the Document class?

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

        document.tokenize()

    end = time.time()
    total_time = end - start
    print(f"Extracting tokens finished in ~{round(total_time, 4)}s")
    print(f"Encountered {errors} errors.\n")
    return documents


def process_tokens(documents: [Document]) -> [Document]:
    """
    TODO: Decide if we want to do this in the __init__ function of the Document class?
    TODO: Grouping together multiple for loops through the documents could improve performance.

    :param documents:
    :return:
    """
    print("Processing tokens")
    for document in documents:
        document.apply_lower_case()
    return documents


def generate_bag_of_tokens(documents: [Document]) -> {}:
    print("Generating bag of tokens.")
    bag_of_tokens = {}
    counter = 0
    for document in documents:
        tokens = document.tokens
        for token in tokens:
            if token not in bag_of_tokens:
                bag_of_tokens[token] = counter
                counter += 1

    return bag_of_tokens


def vectorize_documents(documents: [Document], vectorize_model: _FastText) -> []:
    """
    Converts the tokens into a number that can be used for training a neural network.
    TODO: This should be grouped with the other calls as well

    :param documents:
    :param vectorize_model:
    :return:
    """
    print("Vectorizing documents")
    start = time.time()

    counter = 0
    percentage = 0
    for document in documents:
        counter += 1

        if counter % round(len(documents) / 100) == 0:
            percentage += 1
            counter = 0
            print(f"Vectorizing documents: {percentage}%")
        document.vectorize_tokens(vectorize_model)

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
    end = time.time()
    total_time = end - start
    print(f"Extracting tokens finished in ~{round(total_time, 4)}s")
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

        percentage = 0
        counter = 0
        for document in documents:
            counter += 1
            if counter % round(len(documents) / 100) == 0:
                percentage += 1
                counter = 0
                print(f"Saving documents: {percentage}%")

            file.write("\n")

            content = document.serialize()
            file.write(content)

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
