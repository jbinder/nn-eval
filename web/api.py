import importlib
import os

import flask
import numpy as np
from flask import request, jsonify
from common.csv_data_provider import CsvDataProvider

from normalizer.tools import get_normalizer

config = None
network = None
normalizer = None
app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['POST'])
def home():
    global network, normalizer
    content = request.json
    data = content['input']
    data = normalizer.normalize(np.array(data).reshape(1, -1))[0]
    result = network.predict(data)
    return jsonify({'result': result.tolist()})


def get_network(network_name: str):
    name = network_name.lower()
    module = importlib.import_module(f"networks.{name}.{name}_network")
    network_class = getattr(module, name.capitalize() + "Network")
    return network_class()


def get_config():
    return {
        'ip': os.environ.get('NNE_IP', None),
        'port': int(os.environ.get('NNE_PORT', 0)),
        'network': os.environ.get('NNE_NETWORK', 'pytorch'),
        'normalizer': os.environ.get('NNE_NORMALIZER', 'Identity'),
        'data_x': os.environ.get('NNE_DATA_X', None),
        'data_y': os.environ.get('NNE_DATA_Y', None),
        'model': os.environ.get('NNE_MODEL_FILE', 'model.raw'),
    }


def init():
    global config, network, normalizer
    config = get_config()
    network = get_network(config['network'])
    has_data = config['data_x'] is not None and config['data_y'] is not None
    data = CsvDataProvider().get_data_from_file(config['data_x'], config['data_y'], 1) if has_data else None
    fit_data = np.concatenate((data['train'][0], data['valid'][0]), 0) if data is not None else np.array([0])
    normalizer = get_normalizer(config['normalizer'], fit_data)
    network.load(config['model'])


init()

if __name__ == '__main__':
    app.run(host=config['ip'], port=config['port'] if config['port'] > 0 else None)
