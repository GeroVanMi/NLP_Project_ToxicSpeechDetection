import csv
import time
from os import PathLike

import nltk

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
    TODO: This takes very long, let's cache the result of this somehow.
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
            document.append(nltk.tokenize.word_tokenize(document[COMMENT_TEXT_INDEX]))
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
    for document in documents:
        tokens = document[TOKENS_INDEX]
        document[TOKENS_INDEX] = [token.lower() for token in tokens]
    return documents


def generate_bag_of_tokens(documents: []) -> {}:
    bag_of_tokens = {}
    counter = 0
    for document in documents:
        tokens = document[TOKENS_INDEX]
        for token in tokens:
            if token not in bag_of_tokens:
                bag_of_tokens[token] = counter
                counter += 1

    return bag_of_tokens


def vectorize_tokens(documents: [], bag_of_tokens: {}) -> []:
    """
    Converts the tokens into a number that can be used for training a neural network.
    TODO: This approach is a bit flawed, since if we encounter a word that isn't in the bag of tokens, it probably
          crashes? Maybe we should delete words that don't appear in the bag of tokens?

    :param documents:
    :param bag_of_tokens:
    :return:
    """
    for document in documents:
        tokens = document[TOKENS_INDEX]
        document.append([bag_of_tokens[token] for token in tokens])
    return documents


def limit_documents(documents: [], limit=10) -> {}:
    return documents[0:limit]


def print_documents(documents: [], limit=10) -> None:
    documents = limit_documents(documents, limit)
    for document in documents:
        print(document)


def main():
    file_path = '../data/kaggle/train.csv'
    documents = read_file(file_path)

    # Only for testing
    documents = limit_documents(documents, 100)

    documents = extract_tokens(documents)
    documents = process_tokens(documents)
    bag_of_tokens = generate_bag_of_tokens(documents)

    documents = vectorize_tokens(documents, bag_of_tokens)

    print_documents(documents)
    print(bag_of_tokens)


if __name__ == '__main__':
    main()
