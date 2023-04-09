from Document import Document
import random


def is_toxic(document):
    """
    :param Document document:
    :return:
    """
    return document.is_toxic == 1


def is_not_toxic(document):
    """
    :param Document document:
    :return:
    """
    return document.is_not_toxic == 1


def oversample(documents):
    """
    :param list[Document] documents:
    :return list[Document]:
    """
    toxic_documents = list(filter(is_toxic, documents))
    non_toxic_documents = list(filter(is_not_toxic, documents))
    total_non_toxic = len(non_toxic_documents)

    counter = 0
    while len(toxic_documents) < total_non_toxic:
        toxic_documents.append(toxic_documents[counter])
        counter += 1

    documents = non_toxic_documents + toxic_documents
    random.shuffle(documents)
    # TODO: Maybe this should happen somewhere else?
    return documents
