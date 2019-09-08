import importlib
import os

import flask
from flask import request, jsonify

config = None
network = None
app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['POST'])
def home():
    content = request.json
    # TODO: normalize
    result = network.predict(content['input'])
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
        'model': os.environ.get('NNE_MODEL_FILE', 'model.raw'),
    }


def init():
    global config, network
    config = get_config()
    network = get_network(config['network'])
    network.load(config['model'])


init()

if __name__ == '__main__':
    app.run(host=config['ip'], port=config['port'] if config['port'] > 0 else None)
