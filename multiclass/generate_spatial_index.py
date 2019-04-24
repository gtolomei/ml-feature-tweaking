#!/usr/bin/env python3
# encoding: utf-8

"""
generate_spatial_index.py

Created by Gabriele Tolomei on 2019-04-24.
"""

import sys
import argparse
import logging
import gzip
import multiprocessing as mp
import numpy as np

from sklearn.externals import joblib
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree

from functools import partial


def configure_logging(level=logging.INFO):
    """
    Logging setup
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    LOGGING_FORMAT = '%(asctime)-15s *** %(levelname)s [%(filename)s:%(lineno)s - %(funcName)s()] *** %(message)s'
    formatter = logging.Formatter(LOGGING_FORMAT)

    # log to stdout console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # log to file
    file_handler = logging.FileHandler(
        filename="./generate_spatial_index.log", mode="w")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_options(cmd_args=None):
    """
    Parse command line arguments
    """
    cmd_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""Generate a spatial index data structure from all the k-labelled transformation (i.e., one index for each class label k).""")
    cmd_parser.add_argument(
        'transformations_filename',
        help="""Path to the file containing the serialized transformations.""",
        type=str)
    cmd_parser.add_argument(
        'output_filename',
        help="""Path to the output file, which will contain the spatial index.""",
        type=str)
    cmd_parser.add_argument(
        '-t',
        '--type',
        default='kd-tree',
        const='kd-tree',
        nargs='?',
        choices=['kd-tree', 'ball-tree'],
        help="""Type of spatial index to be generated (default: %(default)s)""")
    args = cmd_parser.parse_args(cmd_args)

    options = {}
    options['transformations_filename'] = args.transformations_filename
    options['output_filename'] = args.output_filename
    options['type'] = args.type

    return options


def load_transformations(transformations_filename):
    with open(transformations_filename, 'rb') as input_file:
        return joblib.load(input_file)


def save_spatial_indices(index, output_filename):
    with gzip.GzipFile(output_filename + '.gz', 'wb') as output_file:
        joblib.dump(index, output_file)


def map_compute_spatial_index(instance, index_type=KDTree):

    logger = logging.getLogger(__name__)

    label, X = instance

    logger.info(
        "Computing {} for all transformations with label k = {}".format(index_type.__name__, label))

    return (label, index_type(X))


def main(options):
    logger = configure_logging(level=logging.INFO)

    # Load the serialized transformations previously computed
    logger.info("==> Loading serialized transformations from `{}`".format(
        options['transformations_filename']))
    transformations = load_transformations(options['transformations_filename'])

    logger.info("*************** Transformations Info ***************")
    for k in transformations:
        logger.info("n. of transformations for class label k = {}: {}".format(
            k, len(transformations[k])))
    logger.info("******************************************")

    logger.info("==> Generating `{}` spatial index".format(options['type']))

    index_type = None

    if options['type'] == 'kd-tree':
        index_type = KDTree
    if options['type'] == 'ball-tree':
        index_type = BallTree

    # Setting up multiple processes (i.e., workers)
    logger.info("==> Creating the pool of {} workers".format(mp.cpu_count()))
    pool = mp.Pool(mp.cpu_count())
    logger.info("==> Preparing the input to be sent to each worker of the pool")

    # dictionary of spatial indices, where each class label is a key
    # and a tree (e.g., KDTree or BallTree) is created using the instances (i.e., transformations) generated from the model's internals
    index = {}
    # list of inputs sent to each worker [(input_w1), ..., (input_wm)]
    inputs = []
    # loop through every label
    for label in transformations:
        logger.info(
            "Generate `{}` spatial index data structure from {}-labelled transformations".format(options['type'], label))
        # check if there exists at least one valid instance
        if transformations[label] and len(transformations[label]) > 0:
            # stack (vertically) all the instances (i.e., numpy arrays) which represent k-labelled transformations
            X = np.vstack(transformations[label])
            logger.info(
                "Number of instances to be indexed for class label k = {}: {}".format(label, X.shape[0]))
            # if so, just append the portion of the dataset (plus extra arguments) to the list of inputs
            inputs.append((label, X))
        else:
            logger.info(
                "No transformations associated with label k = {}".format(label))

    func = partial(map_compute_spatial_index, index_type=index_type)
    index = dict(pool.map(func, inputs))

    # Save all the spatial indices
    logger.info("==> Saving all the {} spatial index to `{}.gz`".format(
        options['type'], options['output_filename']))
    save_spatial_indices(index, options['output_filename'])


if __name__ == '__main__':
    sys.exit(main(get_options()))
