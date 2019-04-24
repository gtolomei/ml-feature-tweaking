#!/usr/bin/env python3
# encoding: utf-8

"""
generate_kd_trees.py

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
        filename="./generate_kd_trees.log", mode="w")
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
        description="""Generate a KD tree data structure from all the k-labelled transformation (i.e., one KD tree for each class label k).""")
    cmd_parser.add_argument(
        'transformations_filename',
        help="""Path to the file containing the serialized transformations.""",
        type=str)
    cmd_parser.add_argument(
        'output_filename',
        help="""Path to the output file, which will contain the KD tree index.""",
        type=str)
    args = cmd_parser.parse_args(cmd_args)

    options = {}
    options['transformations_filename'] = args.transformations_filename
    options['output_filename'] = args.output_filename

    return options


def load_transformations(transformations_filename):
    with open(transformations_filename, 'rb') as input_file:
        return joblib.load(input_file)


def save_kd_trees(kd_trees, output_filename):
    with gzip.GzipFile(output_filename + '.gz', 'wb') as output_file:
        joblib.dump(kd_trees, output_file)


def map_compute_kd_trees(instance):

    logger = logging.getLogger(__name__)

    label, X = instance

    logger.info(
        "Computing KD-trees for all classes with label k = {}".format(label))

    return ((label), KDTree(X))


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

    # Setting up multiple processes (i.e., workers)
    logger.info("==> Creating the pool of {} workers".format(mp.cpu_count()))
    pool = mp.Pool(mp.cpu_count())
    logger.info("==> Preparing the input to be sent to each worker of the pool")

    # dictionary of KD-trees, where each class label is a key
    # and a KD-tree is created using the instances (i.e., transformations) generated from the model's internals
    kd_trees = {}
    # list of inputs sent to each worker [(input_w1), ..., (input_wm)]
    inputs = []
    # loop through every label
    for label in transformations:
        logger.info(
            "Generate KD-tree data structure from {}-labelled transformations".format(label))
        # check if there exists at least one valid instance
        if transformations[label] and len(transformations[label]) > 0:
            # stack (vertically) all the instances (i.e., numpy arrays) which represent k-labelled transformations
            X = np.vstack(transformations[label])
            logger.info(
                "Number of instances to be indexed with KD-tree for label k = {}: {}".format(label, X.shape[0]))
            # if so, just append the portion of the dataset (plus extra arguments) to the list of inputs
            inputs.append((label, X))
        else:
            logger.info(
                "No transformations associated with label k = {}".format(label))

    kd_trees = dict(pool.map(map_compute_kd_trees, inputs))

    for k in kd_trees:
        print(dir(kd_trees[k]))

    # Extract all the (k-leaved) paths from the loaded tree ensemble
    # logger.info("==> Extracting all paths from the just loaded model ...")
    # paths = enumerate_paths(
    #     model, tmp_filename=options['model_filename'] + '.paths.tmp')

    # for k in model.classes_:
    #     k_tot = 0
    #     for t in range(len(model.estimators_)):
    #         logger.debug(
    #             "Number of {}-leaved paths of tree ID #{}: {}".format(k, t, len(paths[k][t])))
    #         k_tot += len(paths[k][t])

    #     logger.info("Total number of {}-leaved paths: {}".format(k, k_tot))

    # if os.path.isfile(options['model_filename'] + '.paths.tmp'):
    #     # Clear temporary path file
    #     logger.info("==> Cleaning up temporary path file `{}`".format(
    #         options['model_filename'] + '.paths.tmp'))
    #     os.remove(options['model_filename'] + '.paths.tmp')

    # # Save all the k-leaved paths to disk
    # logger.info("==> Saving all the extracted paths to `{}.gz`".format(
    #     options['output_filename']))
    # dump_paths(paths, options['output_filename'])


if __name__ == '__main__':
    sys.exit(main(get_options()))
