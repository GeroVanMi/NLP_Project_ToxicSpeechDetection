from os.path import exists
 
import matplotlib.pyplot as plt
import pandas as pd


def generate_score_visualization(root_path, model_name):
    """
    :param PathLike[str] root_path:
    :param str model_name:
    :return:
    """
    model_path = f'{root_path}/logs/{model_name}'
    save_path = f'{model_path}/model_scores.png'
    if exists(save_path):
        return


    model_scores_df = pd.read_csv(f'{model_path}/model_scores.csv', sep=';')
    
    model_scores_df.head()

    plt.plot('total_samples', 'training_accuracy', data=model_scores_df, label='Training Accuracy')
    plt.plot('total_samples', 'validation_accuracy', data=model_scores_df, label='Validation Accuracy')
    plt.plot('total_samples', 'test_accuracy', data=model_scores_df, label='Testing Accuracy')

    plt.legend()
    plt.title("Model: " + model)
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Samples')

    plt.savefig(save_path, bbox_inches='tight')


# plt.show()


if __name__ == '__main__':
    model = 'fully_trained_model2'

    root_path = '../..'

    generate_score_visualization(root_path, model)
