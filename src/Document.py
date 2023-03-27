import csv
from os import PathLike
from nltk.corpus import stopwords

import nltk
import numpy as np


class Document:
    tokens: list[str]
    token_vector: list[int]
    id: str

    is_toxic: int
    is_not_toxic: int
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

        # Binary classification categories
        self.is_toxic = int(toxic)
        self.is_not_toxic = 1 if self.is_toxic == 0 else 0

        # Other categories
        self.severe_toxic = severe_toxic
        self.obscene = obscene
        self.threat = threat
        self.insult = insult
        self.identity_hate = identity_hate

    def tokenize(self):
        self.tokens = nltk.tokenize.regexp_tokenize(self.content, r'[a-zA-Z]+')
        return self

    def remove_stop_words(self):
        new_tokens = []

        english_stopwords = stopwords.words('english')
        for token in self.tokens:
            if token not in english_stopwords:
                new_tokens.append(token)
        self.tokens = new_tokens

        return self

    def apply_lower_case(self):
        self.tokens = [token.lower() for token in self.tokens]
        return self

    def vectorize_tokens(self, bag_of_tokens: dict):
        token_vector = []

        for (_, token) in enumerate(self.tokens):
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
            str(self.is_toxic),
            str(self.severe_toxic),
            str(self.obscene),
            str(self.threat),
            str(self.insult),
            str(self.identity_hate),
            np_separator.join(map(str, self.token_vector)),
        ])

    def one_hot_encode(self, bag_of_tokens: dict[str, int]):
        vector = np.zeros(len(bag_of_tokens))
        for token_index in self.token_vector:
            vector[int(token_index) - 1] = 1
        return vector.astype(int)

    @classmethod
    def deserialize(cls, row: list[str], np_separator=","):
        document = cls(
            row[0],
            "",
            int(row[1]),
            int(row[2]),
            int(row[3]),
            int(row[4]),
            int(row[5]),
            int(row[6]),
        )
        document.token_vector = [int(entry) for entry in row[7].split(np_separator)]

        return document


def load_documents(file_path: str | PathLike[str]) -> list[Document]:
    documents = []
    with open(file_path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        for (index, row) in enumerate(reader):
            if index != 0:
                documents.append(Document.deserialize(row))

    return documents


def limit_documents(documents: list, limit=10) -> list[Document]:
    return documents[0:limit + 1]


def print_documents(documents: list, limit=10) -> None:
    documents = limit_documents(documents, limit)
    for document in documents:
        print(document)


def extract_training_data(documents: list[Document], bag_of_tokens: dict[str, int]):
    x = []
    y = []

    for document in documents:
        x.append(document.one_hot_encode(bag_of_tokens))

        y.append(np.array([
            document.is_not_toxic,
            document.is_toxic,
        ]))

    return np.array(x, dtype=int), np.array(y, dtype=int)
