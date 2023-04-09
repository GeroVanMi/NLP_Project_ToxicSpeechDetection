import os

from keras import Model


class Log:
    def __init__(self, root_path, dir_name):
        self.log_path = f'{root_path}/logs/{dir_name}/'
        os.makedirs(self.log_path)
        os.makedirs(self.log_path + "/data/")

        self.initialize_scores_csv()

    def initialize_scores_csv(self):
        with open(f'{self.log_path}/model_scores.csv', mode='w') as file:
            csv = [
                'total_samples',
                'training_loss',
                'training_accuracy',
                'validation_loss',
                'validation_accuracy',
                'test_loss',
                'test_accuracy',
                'test_f1_score',
            ]

            file.write(";".join(csv) + "\n")

    def log_data_processing(self, message):
        with open(f'{self.log_path}/data_processing.log', mode='a') as file:
            file.write(message + "\n")

    def log_model_structure(self, model):
        model_json = model.to_json()
        with open(f'{self.log_path}/model_structure.json', mode='a') as file:
            file.write(model_json + "\n")

    def log_model_scores(self, total_samples, training_scores, test_scores):
        with open(f'{self.log_path}/model_scores.csv', mode='a') as file:
            csv = [
                total_samples,
                training_scores['loss'],
                training_scores['accuracy'],
                training_scores['validation_loss'],
                training_scores['validation_accuracy'],
                test_scores['loss'],
                test_scores['accuracy'],
                test_scores['f1_score'],
            ]

            file.write(";".join(map(str, csv)) + "\n")

    def log_model_evaluation(self, message):
        with open(f'{self.log_path}/model_evaluation.log', mode='a') as file:
            file.write(message + "\n")

    def save_model(self, model: Model):
        model.save(self.log_path + 'toxic_detection_model')

    def log_exception(self, exception):
        with open(f'{self.log_path}/runtime_error.txt', mode='a') as file:
            file.write(str(exception))
