def limit_documents(documents: [], limit=10) -> {}:
    return documents[0:limit]


def print_documents(documents: [], limit=10) -> None:
    documents = limit_documents(documents, limit)
    for document in documents:
        print(document)
