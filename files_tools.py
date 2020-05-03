#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
All functions to read, write json files or csv files
"""

from __future__ import absolute_import

import io
import json
import logging.handlers
import os
import shutil

PYTHON_LOGGER = logging.getLogger(__name__)
if not os.path.exists("log"):
    os.mkdir("log")
HDLR = logging.handlers.TimedRotatingFileHandler("log/files_tools.log",
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


def read_json_file(json_file_abs_path):
    """
    Read a json file in utf-8
    :param json_file_abs_path: (string) path to the json to read
    :return: (dic) dic of the json file
    """
    with io.open(json_file_abs_path, "r", encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data


def save_json_file(json_dic, file_name, indent=None):
    """
    Save dic to json file
    :param json_dic: (dict) dictionary need to be save
    :param file_name: (string) name of the file
    :param indent: (int) if not None add indent
    """
    with io.open(file_name, "w", encoding="utf-8") as f:
        if indent is None:
            json.dump(json_dic, f, ensure_ascii=False)
        else:
            json.dump(json_dic, f, indent=indent, ensure_ascii=False)
