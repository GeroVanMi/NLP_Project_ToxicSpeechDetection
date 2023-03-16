import nltk
import numpy as np


class Document:
    tokens: [str]
    token_vectors: np.ndarray
    id: str

    toxic: int
    severe_toxic: int
    obscene: int
    threat: int
    insult: int
    identity_hate: int

    def __init__(
            self,
            id: str,
            content: str,
            toxic: int,
            severe_toxic: int,
            obscene: int,
            threat: int,
            insult: int,
            identity_hate: int,
    ):
        self.id = id
        self.content = content
        self.toxic = toxic
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

    def vectorize_tokens(self, vectorize_model):
        token_vectors = np.ndarray(shape=(len(self.tokens), 100))

        for (index, token) in enumerate(self.tokens):
            token_vectors[index] = vectorize_model.get_word_vector(token)

        self.token_vectors = token_vectors

        return self

    def serialize(self, seperator=";", np_seperator=" ") -> str:
        """
        Converts the document into a string, in order to save it to a file.
        TODO: Handle error for when this function is called too early.

        :param seperator: Which string to use to differentiate between the fields of the document
        :param np_seperator:  Which string to use to differentiate between the values in the token_vectors numpy array.
        :return:
        """
        return seperator.join([
            self.id,
            self.toxic,
            self.severe_toxic,
            self.obscene,
            self.threat,
            self.insult,
            self.identity_hate,
            " ".join(list(self.token_vectors.reshape(-1).astype(str)))
        ])
