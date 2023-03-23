import json


def read_bag_of_tokens(file_path: str) -> dict:
    with open(file_path, mode='r') as file:
        return json.load(file)
