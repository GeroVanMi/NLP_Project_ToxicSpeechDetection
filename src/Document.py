import csv
from os import PathLike

import nltk
import numpy as np


class Document:
    tokens: [str]
    token_vector: [int]
    id: str

    is_toxic: int
    severe_toxic: int
    obscene: int
    threat: int
    insult: int
    identity_hate: int

    def __init__(
            self,
            document_id: str,
            content: str,
            toxic: int,
            severe_toxic: int,
            obscene: int,
            threat: int,
            insult: int,
            identity_hate: int,
    ):
        self.id = document_id
        self.content = content
        self.is_toxic = toxic
        self.severe_toxic = severe_toxic
        self.obscene = obscene
        self.threat = threat
        self.insult = insult
        self.identity_hate = identity_hate

    def tokenize(self):
        self.tokens = nltk.tokenize.regexp_tokenize(self.content, r'[a-zA-Z]+')
        return self

    def apply_lower_case(self):
        """
        TODO: This could be applied to the content immediately for performance improvement?
        :return:
        """
        self.tokens = [token.lower() for token in self.tokens]
        return self

    def vectorize_tokens(self, bag_of_tokens: {}) -> [int]:
        token_vector = []

        for (index, token) in enumerate(self.tokens):
            if token in bag_of_tokens:
                token_vector.append(bag_of_tokens[token])

        self.token_vector = token_vector

        return self

    def serialize(self, separator=";", np_separator=",") -> str:
        """
        Converts the document into a string, in order to save it to a file.
        TODO: Handle error for when this function is called too early.

        :param separator: Which string to use to differentiate between the fields of the document
        :param np_separator:  Which string to use to differentiate between the values in the token_vectors numpy array.
        :return:
        """
        return separator.join([
            self.id,
            self.is_toxic,
            self.severe_toxic,
            self.obscene,
            self.threat,
            self.insult,
            self.identity_hate,
            np_separator.join(map(str, self.token_vector)),
        ])

    def one_hot_encode(self, bag_of_tokens: {str: int}):
        vector = np.zeros(len(bag_of_tokens))
        for token_index in self.token_vector:
            vector[int(token_index) - 1] = 1
        return vector.astype(int)

    @classmethod
    def deserialize(cls, row: [str], np_separator=","):
        document = cls(
            row[0],
            "",
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            row[6],
        )
        document.token_vector = [int(entry) for entry in row[7].split(np_separator)]

        return document


def load_documents(file_path: str | PathLike[str]) -> [Document]:
    documents = []
    with open(file_path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        for (index, row) in enumerate(reader):
            if index != 0:
                documents.append(Document.deserialize(row))

    return documents


def limit_documents(documents: [], limit=10) -> {}:
    return documents[0:limit + 1]


def print_documents(documents: [], limit=10) -> None:
    documents = limit_documents(documents, limit)
    for document in documents:
        print(document)


def extract_training_data(documents: [Document], bag_of_tokens: {str: int}):
    x = []
    y = []

    for document in documents:
        x.append(document.one_hot_encode(bag_of_tokens))
        y.append(np.array([
            document.is_toxic,
            document.severe_toxic,
            document.obscene,
            document.threat,
            document.insult,
            document.identity_hate
        ]))

    return np.array(x, dtype=int), np.array(y, dtype=int)
