import keras
from keras import backend
from keras.layers import Dense

from Document import load_documents, extract_training_data
from bag_of_tokens import read_bag_of_tokens

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/", methods=['POST'])
@cross_origin()
def run_model():
    request_body = request.json

    if not request_body['modelName']:
        return jsonify({
            "error": "You need to supply a model name"
        }), 400

    if not request_body['modelName']:
        return jsonify({
            "error": "You need to supply a document index"
        }), 400

    root_path = '../../logs'
    model_name = request_body['modelName']
    document_index = int(request_body['documentIndex'])
    model_path = f'{root_path}/{model_name}/toxic_detection_model'

    try:
        model = keras.models.load_model(model_path)
        model.summary()

        documents = load_documents(f'{root_path}/{model_name}/data/train.csv')

        documents = documents[document_index:document_index + 1]

        bag_of_tokens = read_bag_of_tokens(f'{root_path}/{model_name}/data/bag_of_words.json')
    except (FileNotFoundError, OSError):
        return jsonify({
            "error": "Model could not be found."
        }), 404

    # TODO: If we want to be able to define the sentence ourself we have to
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


if __name__ == '__main__':
    app.run(debug=True)
