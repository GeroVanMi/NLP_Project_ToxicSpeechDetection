import time

from alive_progress import alive_bar

from Document import Document
from Log import Log
from Settings import Settings


def extract_tokens(documents, logger=None):
    start = time.time()

    with alive_bar(len(documents), title="Extracting tokens") as update_bar:
        for document in documents:
            update_bar()
            document.tokenize()
    print()

    end = time.time()
    total_time = end - start
    if logger is not None:
        logger.log_data_processing(f"Extracting tokens:{total_time}")

    return documents


def apply_lowercase(documents):
    with alive_bar(len(documents), title="Applying lowercase") as update_bar:
        for document in documents:
            update_bar()
            document.apply_lower_case()
    print()

    return documents


def remove_stopwords(documents, logger=None):
    start = time.time()

    with alive_bar(len(documents), title="Removing stopwords") as update_bar:
        for document in documents:
            update_bar()
            document.remove_stop_words()
    print()

    end = time.time()
    total_time = end - start
    if logger:
        logger.log_data_processing(f"Extracting tokens:{total_time}")

    return documents


def process_documents(documents, logger, settings):
    documents = extract_tokens(documents, logger)

    if settings.lower_case:
        documents = apply_lowercase(documents)

    if settings.remove_stop_words:
        documents = remove_stopwords(documents, logger)

    return documents
