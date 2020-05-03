#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""


Usage:
  model_test.py <model_name>
  model_test.py -h | --help

Options:
  -h --help     Show this screen.
  <model_name>      Path to config file
"""

from __future__ import absolute_import

import logging.handlers
import os

import cv2
from docopt import docopt
from imutils import paths
from tensorflow.keras.models import load_model
import numpy as np

from creat_train_dataset import ResizePreprocessor, ImageToArrayPreprocessor
from files_tools import read_json_file

PYTHON_LOGGER = logging.getLogger(__name__)
if not os.path.exists("log"):
    os.mkdir("log")
HDLR = logging.handlers.TimedRotatingFileHandler("log/model_test.log",
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
DATASET = os.path.join(FOLDER_ABSOLUTE_PATH, "test_dataset")

args = docopt(__doc__)

model_name = args["<model_name>"]

d = read_json_file(f"{model_name}.json")
label_list = d["labels"]

image_paths = list(paths.list_images(DATASET))

model = load_model(f"{model_name}.h5")
rp = ResizePreprocessor(d["img_dim"], d["img_dim"])
ita = ImageToArrayPreprocessor()

print("Enter 'esc' to exit or 'n' for next image")

for image_path in image_paths:

    test_img = cv2.imread(image_path)
    test_input = rp.preprocess(test_img)
    test_input = ita.preprocess(test_input)
    test_input = np.array([test_input.astype("float") / 255.0])

    res = model.predict(test_input)
    label = label_list[res.argmax(axis=1)[0]]
    cv2.putText(test_img, f"Label: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Classification", test_img)
    end = False
    while True:
        k = cv2.waitKey(33)
        # Esc key to stop
        if k == 27:
            end = True
            break
        if k == 110:
            break
