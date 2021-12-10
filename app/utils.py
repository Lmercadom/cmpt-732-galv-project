#
# author: vva12 Vishakha
#

import logging
import sys
import os

import numpy as np
import pandas as pd


def create_logger(consolelevel=logging.INFO):
    """
    create a universal logger object to be used across modules
    :param consolelevel: logging level
    :return: logger object
    """
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(consolelevel)
    logger = logging.getLogger()
    logger.addHandler(console_handler)

    return logger


def get_word_vec(attributes):
    """

    :return: available_attr_dict ['str':np.array]
    """
    embeddings_index = {}
    logging.info("Loading GloVe 50d")
    with open('glove.6B.50d.txt', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in attributes:
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

    return embeddings_index

def read_business_file(path, lat_deg, long_deg):
    """
    read the business data file. The idea is to read only
    those businesses that have the same degree of lat or
    long.
    :param path: str: path to data file
    :param lat_deg: user lat in deg
    :param long_deg: user long in deg
    :return: pd.DataFrame
    """
    records = []
    files = os.listdir(path)
    files = [file for file in files if file.endswith(".csv")]
    logging.info(f"Reading files: {', '.join(files)}")
    for file in files:
        logging.debug(f"Reading {os.path.join(path, file)}")
        with open(os.path.join(path, file), 'r') as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    headers = line.strip().split("^")
                    continue
                coords = line.split("^")[5:7]
                coords = [int(coord) for coord in coords]
                if lat_deg in coords or long_deg in coords:
                    # lat/long deg match read the record
                    records.append(line.split("^"))
    df = pd.DataFrame(records, columns=headers)
    # add near flag for downstream compatibility
    df['near'] = True
    return df