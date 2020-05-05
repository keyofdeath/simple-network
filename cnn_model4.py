#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import logging.handlers
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

from creat_train_dataset import ImageCreatTrainDataset
from files_tools import save_json_file
from alexnet import AlexNet

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
IMG_DIM = 128
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.01
LAYER_DIM = IMG_DIM * IMG_DIM * 3
LAYERS = [
    LAYER_DIM * 2,
    LAYER_DIM,
    LAYER_DIM // 2
]

# Dataset
dataset = ImageCreatTrainDataset(DATASET, IMG_DIM)
dataset.load_dataset()
train_x, train_y = dataset.get_train_data()
test_x, test_y = dataset.get_test_data()
labels, nb_labels = dataset.get_labels()

model = AlexNet.build(width=IMG_DIM, height=IMG_DIM, depth=3, classes=2, reg=0.0002)

loss = "categorical_crossentropy" if nb_labels > 2 else "binary_crossentropy"

sgd = SGD(lr=LEARNING_RATE, decay=LEARNING_RATE / EPOCHS, momentum=0.9)
model.compile(loss=loss, optimizer=sgd, metrics=["accuracy"])
H = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=EPOCHS, batch_size=BATCH_SIZE)

# evaluate the network

PYTHON_LOGGER.info("Evaluating network")
predictions = model.predict(test_x, batch_size=BATCH_SIZE)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=labels))

# plot the training loss and accuracy
range_plot = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(range_plot, H.history["loss"], label="train_loss", color='red')
plt.plot(range_plot, H.history["val_loss"], label="val_loss", color='green')
plt.plot(range_plot, H.history["accuracy"], label="train_acc", color='blue')
plt.plot(range_plot, H.history["val_accuracy"], label="val_acc", color='pink')
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

save_json_file({"img_dim": IMG_DIM, "labels": labels}, "model_4.json")
model.save("model_4.h5")
