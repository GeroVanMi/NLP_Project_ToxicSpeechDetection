import argparse
import time

from Log import Log
from Settings import Settings
from data_processing.data_preprocessing import process_data
from model.model_evaluation import evaluate_model
from model.model_training import train_model


def run_pipeline():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--limit')
    parser.add_argument('--epochs')
    parser.add_argument('--batch-size')
    parser.add_argument('-n', '--name')
    arguments = parser.parse_args()

    root_path = '..'

    if arguments.name:
        log_name = arguments.name
    else:
        log_name = str(round(start_time))

    logger = Log(root_path, log_name)
    settings = Settings() \
        .enable_oversample()

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
            print(f"Data limit \"{arguments.epochs}\" could not be parsed. Ignoring.")

    batch_size = 500
    if arguments.batch_size is not None:
        try:
            batch_size = int(arguments.batch_size)
        except ValueError:
            print(f"Data limit \"{arguments.batch_size}\" could not be parsed. Ignoring.")

    process_data(root_path=root_path, logger=logger, settings=settings, limit=data_limit)
    train_model(logger=logger, limit=data_limit, number_of_epochs=epochs, desired_batch_size=batch_size)
    evaluate_model(logger=logger, limit=data_limit)


if __name__ == '__main__':
    run_pipeline()