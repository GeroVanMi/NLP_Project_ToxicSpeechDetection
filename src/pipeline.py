import argparse
import time

from Log import Log
from data_processing.data_preprocessing import process_data
from model.model_evaluation import evaluate_model
from model.model_training import train_model


def run_pipeline():
    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--limit')

    root_path = '..'

    logger = Log(root_path, str(round(start_time)))

    arguments = parser.parse_args()

    data_limit = None
    if arguments.limit is not None:
        try:
            data_limit = int(arguments.limit)
        except ValueError:
            print(f"Data limit \"{arguments.limit}\" could not be parsed. Ignoring.")

    process_data(root_path=root_path, logger=logger, limit=data_limit)
    train_model(logger=logger, limit=data_limit)
    evaluate_model(logger=logger, limit=data_limit)


if __name__ == '__main__':
    run_pipeline()
