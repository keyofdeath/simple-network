#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import logging.handlers
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from creat_train_dataset import ImageCreatTrainDataset
from files_tools import save_json_file

PYTHON_LOGGER = logging.getLogger(__name__)
if not os.path.exists("log"):
    os.mkdir("log")
HDLR = logging.handlers.TimedRotatingFileHandler("log/model1.log",
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
DATASET = os.path.join(FOLDER_ABSOLUTE_PATH, "dog_cat_dataset")
IMG_DIM = 224
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Dataset
dataset = ImageCreatTrainDataset(DATASET, IMG_DIM)
dataset.load_dataset()
train_x, train_y = dataset.get_train_data()
test_x, test_y = dataset.get_test_data()
labels, nb_labels = dataset.get_labels()

loss = "categorical_crossentropy" if nb_labels > 2 else "binary_crossentropy"

baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(IMG_DIM, IMG_DIM, 3)))

for layer in baseModel.layers:
    layer.trainable = False

headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)

# add a softmax layer
headModel = Dense(nb_labels, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

sgd = SGD(lr=LEARNING_RATE)
model.compile(loss=loss, optimizer=sgd, metrics=["accuracy"])
H = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=EPOCHS, batch_size=BATCH_SIZE)

# evaluate the network

PYTHON_LOGGER.info("Evaluating network")
predictions = model.predict(test_x, batch_size=BATCH_SIZE)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=labels))

save_json_file({"img_dim": IMG_DIM, "labels": labels}, "model_4.json")
model.save("model_4.h5")
