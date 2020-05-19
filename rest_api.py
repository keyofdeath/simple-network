#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Usage:
  rest_api.py <model>

Options:
  -h --help      Show this screen.
  <model>        Teachable machine model path
"""

from __future__ import absolute_import
import os
import logging.handlers
from flask import request, jsonify, Flask
from waitress import serve
from threading import Lock
from docopt import docopt
import tensorflow as tf
import numpy as np
import cv2

PYTHON_LOGGER = logging.getLogger(__name__)
if not os.path.exists("log"):
    os.mkdir("log")
HDLR = logging.handlers.TimedRotatingFileHandler("log/rest_api.log",
                                                 when="midnight", backupCount=60)
STREAM_HDLR = logging.StreamHandler()
FORMATTER = logging.Formatter("%(asctime)s %(filename)s [%(levelname)s] %(message)s")
HDLR.setFormatter(FORMATTER)
STREAM_HDLR.setFormatter(FORMATTER)
PYTHON_LOGGER.addHandler(HDLR)
PYTHON_LOGGER.addHandler(STREAM_HDLR)
PYTHON_LOGGER.setLevel(logging.DEBUG)

# Absolute path to the folder location of this python file
FOLDER_ABSOLUTE_PATH = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))

args = docopt(__doc__)

app = Flask(__name__)
app.secret_key = "zefz@@#'rega5zerg+6e5rzeafz"
MUTEX = Lock()
SIZE = (224, 224)

# Load the model
model = tf.keras.models.load_model(os.path.join(args["<model>"], "keras_model.h5"))
with open(os.path.join(args["<model>"], "labels.txt")) as f:
    # Remove \n and remove the index next to the label name: <index> <label name>
    labels = [' '.join(l.replace('\n', '').split(' ')[1:]) for l in f]


def process_input(json_array):

    frame = np.array(json_array, dtype=np.uint8)
    frame = cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA)
    normalized_image_array = (frame.astype(np.float32) / 127.0) - 1
    model_input = np.expand_dims(normalized_image_array, axis=0)
    with MUTEX:
        prediction = model.predict(model_input)

    return labels[prediction.argmax(axis=1)[0]]


@app.route('/', methods=["POST"])
def message_interact():
    try:
        PYTHON_LOGGER.info("Request")
        json_data = request.get_json()
        img = json_data.get("img")
        res = process_input(img)
        PYTHON_LOGGER.info(f"Label: {res}")
        return jsonify({"label": res})
    except Exception as e:
        PYTHON_LOGGER.error("Error to get image: {}".format(e))


# Start flask server
serve(app, host="0.0.0.0", port=8888)
