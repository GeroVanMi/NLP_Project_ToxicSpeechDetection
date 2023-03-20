import os
from os import PathLike

from keras import Model
from keras.callbacks import History


class Log:
    def __init__(self, root_path: str | PathLike[str], dir_name: str):
        self.log_path = f'{root_path}/logs/{dir_name}/'
        os.makedirs(self.log_path)
        os.makedirs(self.log_path + "/data/")

    def log_data_processing(self, message: str):
        with open(f'{self.log_path}/data_processing.log', mode='a') as file:
            file.write(message + "\n")

    def log_model_structure(self, model: Model):
        model_json = model.to_json()
        with open(f'{self.log_path}/model_structure.json', mode='a') as file:
            file.write(model_json + "\n")

    def log_model_history(self, history: History):
        history = history.history
        with open(f'{self.log_path}/training_loss.log', mode='a') as file:
            file.write(str(history['loss']) + "\n")

        with open(f'{self.log_path}/training_accuracy.log', mode='a') as file:
            file.write(str(history['accuracy']) + "\n")

        with open(f'{self.log_path}/validation_loss.log', mode='a') as file:
            file.write(str(history['val_loss']) + "\n")

        with open(f'{self.log_path}/validation_accuracy.log', mode='a') as file:
            file.write(str(history['val_accuracy']) + "\n")

    def save_model(self, model: Model):
        model.save(self.log_path + 'toxic_detection_model')
