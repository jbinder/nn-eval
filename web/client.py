import os

import flask
import requests
from flask import request, render_template, current_app, flash, redirect, url_for

config = None
normalizer = None
app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.secret_key = "secret_key"


@app.route('/', methods=['GET'])
def home():
    return render_template("client_index.html")


@app.route('/predict', methods=['POST'])
def predict():
    json = {'input': (request.form['input'].split(','))}
    try:
        response = requests.post(current_app.config.api_url, json=json)
        if response.status_code == requests.codes.ok:
            result = response.json()['result'][0]
            flash(result)
        else:
            flash("Error: " + response.reason)
    except RuntimeError:
        flash("Error: unable to predict :(")
    return redirect(url_for('home'))


def get_config():
    return {
        'ip': os.environ.get('NNEC_IP', None),
        'port': int(os.environ.get('NNEC_PORT', 0)),
        'api_url': os.environ.get('NNEC_API_URL', 0),
    }


if __name__ == '__main__':
    config = get_config()
    app.config.api_url = config['api_url']
    app.run(host=config['ip'], port=config['port'] if config['port'] > 0 else None)
