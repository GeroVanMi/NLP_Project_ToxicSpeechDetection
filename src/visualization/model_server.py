import os

import keras
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from keras import backend
from keras.layers import Dense

from Document import extract_training_data, Document
from Settings import Settings
from bag_of_tokens import read_bag_of_tokens
from data_processing.document_vectorization import vectorize_documents
from data_processing.documents_processing import process_documents

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

root_path = f'{os.path.dirname(os.path.realpath(__file__))}/../../logs'

@app.route("/models", methods=['GET'])
@cross_origin()
def get_model_names():

    model_names = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]

    return jsonify(model_names)


@app.route("/", methods=['POST'])
@cross_origin()
def run_model():
    request_body = request.json
    if request_body['modelName'] is None:
        return jsonify({
            "error": "You need to supply a model name"
        }), 400

    if request_body['documentContent'] is None:
        return jsonify({
            "error": "You need to supply a document content"
        }), 400

    model_name = request_body['modelName']

    settings = Settings()
    settings.enable_lower_case().enable_oversample().enable_stop_word_removal()

    # document_index = int(request_body['documentIndex'])
    model_path = f'{root_path}/{model_name}/toxic_detection_model'

    try:
        model = keras.models.load_model(model_path)
        model.summary()

    except (FileNotFoundError, OSError) as error:
        print(error)
        return jsonify({
            "error": "Model could not be found."
        }), 404

    # documents = load_documents(f'{root_path}/{model_name}/data/train.csv')

    # documents = documents[document_index:document_index + 1]

    document_content = "Fuck you"
    if isinstance(request_body['documentContent'], str):
        document_content = request_body['documentContent']

    documents = [Document(
        "new",
        document_content,
        1
    )]

    bag_of_tokens = read_bag_of_tokens(f'{root_path}/{model_name}/data/bag_of_words.json')

    documents = process_documents(documents, None, settings)
    documents = vectorize_documents(documents, bag_of_tokens)

    x_test, _ = extract_training_data(documents, bag_of_tokens)

    layers = []
    for layer_index in range(len(model.layers)):
        current_layer: Dense = model.layers[layer_index]

        if layer_index == 0:
            weights = current_layer.get_weights()
            # TODO: Basically we could retrieve the 1x1000 weights for a single word and see how it affects the first
            #       hidden layer. (How much of the activation is due to that word.)
            #       And then we could repeat this for the subsequent layers, checking how a single word propagates
            #       through the network.
            kernel_matrix = weights[0]
            bias_vector = weights[1]
            print(kernel_matrix.shape)  # Kernel Matrix
            print(bias_vector.shape)  # Bias vector

        get_layer_output = backend.function([model.layers[0].input], [current_layer.output])
        layer_output = get_layer_output([x_test])[0]
        formatted_output = list(layer_output.reshape(-1))
        layers.append({
            'units': current_layer.units,
            'outputs': list(map(str, formatted_output))
        })

    document = documents[0]
    inverted_bag_of_tokens = {v: k for k, v in bag_of_tokens.items()}

    data = {
        "message": "OK",
        "layers": layers,
        "isToxic": document.is_toxic,
        "documentId": document.id,
        "documentTokens": [inverted_bag_of_tokens[token_index] for token_index in document.token_vector],
        "prediction": 'Not Toxic' if float(layers[-1]['outputs'][0]) > float(layers[-1]['outputs'][1]) else 'Toxic',
    }
    return jsonify(data)
