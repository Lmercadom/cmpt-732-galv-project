import logging
import sys

import numpy as np


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