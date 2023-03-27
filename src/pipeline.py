import argparse
import time

from Log import Log
from Settings import Settings
from data_processing.data_preprocessing import process_data
from model.model_training import train_model


def run_pipeline():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name')  # Run / Log name
    # Run configuration
    parser.add_argument('-l', '--limit')
    parser.add_argument('--epochs')
    parser.add_argument('--batch-size')

    # Processing configuration
    parser.add_argument('-o', '--oversample', action='store_true')
    parser.add_argument('-w', '--remove-stop-words', action='store_true')
    parser.add_argument('-lc', '--lowercase', action='store_true')
    arguments = parser.parse_args()

    root_path = '..'

    if arguments.name:
        log_name = arguments.name
    else:
        log_name = str(round(start_time))

    logger = Log(root_path, log_name)
    settings = Settings()

    if arguments.oversample:
        settings.enable_oversample()

    if arguments.remove_stop_words:
        settings.enable_stop_word_removal()

    if arguments.lowercase:
        settings.enable_lower_case()

    data_limit = None
    if arguments.limit is not None:
        try:
            data_limit = int(arguments.limit)
        except ValueError:
            print(f"Data limit \"{arguments.limit}\" could not be parsed. Ignoring.")

    epochs = 5
    if arguments.epochs is not None:
        try:
            epochs = int(arguments.epochs)
        except ValueError:
            print(f"Epochs \"{arguments.epochs}\" could not be parsed. Ignoring.")

    batch_size = 500
    if arguments.batch_size is not None:
        try:
            batch_size = int(arguments.batch_size)
        except ValueError:
            print(f"Batch size \"{arguments.batch_size}\" could not be parsed. Ignoring.")

    process_data(root_path=root_path, logger=logger, settings=settings, limit=data_limit)
    train_model(logger=logger, limit=data_limit, number_of_epochs=epochs, desired_batch_size=batch_size)


if __name__ == '__main__':
    run_pipeline()
