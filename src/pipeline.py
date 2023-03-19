import argparse

from data_processing.data_preprocessing import process_data
from model.model_training import train_model
from model.model_evaluation import evaluate_model


def run_pipeline():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--process-data', action="store_true")
    parser.add_argument('-t', '--train-model', action="store_true")
    parser.add_argument('-e', '--evaluate-model', action="store_true")
    parser.add_argument('-l', '--limit')

    root_path = '..'

    arguments = parser.parse_args()

    data_limit = None
    if arguments.limit is not None:
        try:
            data_limit = int(arguments.limit)
        except ValueError:
            print(f"Data limit \"{arguments.limit}\" could not be parsed. Ignoring.")

    if arguments.process_data:
        process_data(root_path, data_limit)

    if arguments.train_model:
        train_model(root_path, limit=data_limit)

    if arguments.evaluate_model:
        evaluate_model(root_path, limit=data_limit)


if __name__ == '__main__':
    run_pipeline()
