#!/usr/bin/env python3
# encoding: utf-8

"""
extract_knn.py

Created by Gabriele Tolomei on 2019-04-24.
"""

import sys
import argparse
import logging
import time
import numpy as np
import pandas as pd

from sklearn.externals import joblib


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
        filename="./logs/extract_knn.log", mode="w")
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
        description="""Compute and store efficiently all the epsilon-transformations associated with the paths of an ensemble of decision trees, previously extracted and serialized to disk.""")
    cmd_parser.add_argument(
        'dataset_filename',
        help="""Path to the input file containing the original dataset (features + class labels).""",
        type=str)
    cmd_parser.add_argument(
        'transformations_filename',
        help="""Path to the input file containing all the computed transformations.""",
        type=str)
    cmd_parser.add_argument(
        'output_filename',
        help="""Path to the output file containing all the k-NN.""",
        type=str)
    cmd_parser.add_argument(
        '-n',
        '--n_samples',
        help="""Number of samples to be randomly extracted from the whole dataset, and used as query points.""",
        default=100,
        type=int)
    cmd_parser.add_argument(
        '-k',
        '--knn',
        help="""Number k of nearest neighbors to be computed.""",
        default=1,
        type=int)
    cmd_parser.add_argument(
        '-i',
        '--index_filename',
        help="""Path to the spatial index used to compute k-NN.""",
        type=str)
    cmd_parser.add_argument(
        '-d',
        '--distance',
        default='l2',
        const='l2',
        nargs='?',
        choices=['l1', 'l2', 'cosine'],
        help="""Distance metric used to compute k-NN (default: %(default)s)""")
    args = cmd_parser.parse_args(cmd_args)

    options = {}
    options['dataset_filename'] = args.dataset_filename
    options['transformations_filename'] = args.transformations_filename
    options['n_samples'] = args.n_samples
    options['knn'] = args.knn
    options['output_filename'] = args.output_filename
    options['index_filename'] = args.index_filename
    options['distance'] = args.distance

    return options


######################## Loading Dataset as a Pandas DataFrame object ####


def load_dataset(input_filename, sep="\t", fillna=None):
    """
    This function is responsible for loading the input dataset.
    The internal representation of the dataset is a pandas.DataFrame object

    Args:
        input_filename (str): path to the input file containing the extracted data
        sep (str): character used to delimit the input file (default = "\t")
        fillna (str): replacement string for NA entries (default = None)

    Return:
        dataset (pandas.DataFrame): Pandas Dataframe object representing the input dataset
    """
    logger = logging.getLogger(__name__)

    logger.info("Loading dataset from " + input_filename +
                " into Pandas DataFrame object...")
    logger.info("Inferring compression from " + input_filename + " (if any)")
    compression = None
    if input_filename.split('.')[-1] == 'bz2':
        compression = 'bz2'
    if input_filename.split('.')[-1] == 'gz':
        compression = 'gzip'
    logger.info("Compression = " + str(compression))
    dataset = pd.read_csv(input_filename, sep=sep, compression=compression)
    if fillna:
        logger.info(
            "Replacing missing values (i.e. NA) with \"" + fillna + "\"")
        dataset.fillna(fillna, inplace=True)
    logger.info("Returning the dataset")
    return dataset

##########################################################################


def load_model(model_filename):
    """
    This function loads a model from a dump done via scikit-learn

    Args:
        model_filename (str): path to the file where the model has been serialized and persisted

    Return:
        an instance representing the trained model
    """
    with open(model_filename, 'rb') as model_file:
        return joblib.load(model_file)

##########################################################################


def load_transformations(transformations_filename):
    with open(transformations_filename, 'rb') as input_file:
        return joblib.load(input_file)

##########################################################################


def load_spatial_index(index_filename):
    with open(index_filename, 'rb') as input_file:
        return joblib.load(input_file)


##########################################################################

def compute_distance(x, c):
    return np.linalg.norm(x - c)


##########################################################################


def get_knn_brute_force(X, true_labels, target_labels, ids, transformations, k):

    logger = logging.getLogger(__name__)

    knn = {}
    for i in range(X.shape[0]):
        x = X[i]
        y = true_labels[i]
        x_id = ids[i]
        knn[(x_id, y)] = {}
        for label in target_labels[i]:
            logger.info(
                "Retrieve all the {}-labelled transformations computed previously ...".format(label))
            candidates = transformations[label]
            if candidates and len(candidates) > 0:
                logger.info(
                    "Compute all-pair distances from instance id #{} originally labelled as `{}` to the [{}] {}-labelled candidates...".format(x_id, y, len(candidates), label))
                # compute all-pair distances from this query vector to all the candidates
                distances = [(compute_distance(x, c), c_i) for c_i, c in enumerate(
                    candidates)]
                distances = sorted(distances, key=lambda tup: tup[0])
                knn[(x_id, y)][label] = distances[:k]
            else:
                logger.info(
                    "No {}-labelled candidates available!".format(label))

    return knn


##########################################################################


def get_knn_opt(X, true_labels, target_labels, ids, k, spatial_index):

    logger = logging.getLogger(__name__)

    knn = {}
    for i in range(X.shape[0]):
        x = X[i]
        y = true_labels[i]
        x_id = ids[i]
        knn[(x_id, y)] = {}
        for label in target_labels[i]:
            if label in spatial_index:
                logger.info(
                    "Retrieve the spatial index trained on the {}-labelled transformations computed previously ...".format(label))
                tree = spatial_index[label]
                logger.info(
                    "Retrieve the {}-nearest {}-labelled neighbours to this instance id #{} originally labelled as `{}`".format(k, label, x_id, y))
                nearest_dist, nearest_ind = tree.query(x.reshape(1, -1), k=k)
                knn[(x_id, y)][label] = list(
                    zip(nearest_dist.flatten().tolist(), nearest_ind.flatten().tolist()))

    return knn

##########################################################################


def get_knn(X, true_labels, target_labels, ids, transformations, k=1, spatial_index=None):

    logger = logging.getLogger(__name__)

    if spatial_index:
        logger.info("==> Extract k-NN using efficient spatial index ...")
        return get_knn_opt(X, true_labels, target_labels, ids, k, spatial_index)
    else:
        logger.info("==> Extract k-NN using brute force ...")
        return get_knn_brute_force(X, true_labels, target_labels, ids, transformations, k)


##########################################################################


def get_target_labels(y, labels):
    return np.array([c for c in labels if c != y])


##########################################################################


def main(options):

    logger = configure_logging(level=logging.INFO)

    # Set the default random seed
    np.random.seed(42)

    # Loading dataset
    logger.info("==> Loading dataset from `{}`".format(
        options['dataset_filename']))
    dataset = load_dataset(options['dataset_filename'], sep=",")

    logger.info("Shape of the dataset: {} instances by {} features".format(
        dataset.shape[0], dataset.shape[1] - 1))

    true_labels = sorted(dataset["label"].unique())

    # Extract a random sample of n query points
    logger.info("==> Extracting a random sample (without replacement) of n = {} query points out of the whole dataset".format(
        options['n_samples']))
    sample_ids = np.random.choice(
        dataset.shape[0], options['n_samples'], replace=False)
    sample_dataset = dataset.iloc[sample_ids, :]

    X_query = sample_dataset.iloc[:, 1:].values  # sample feature matrix
    y_query = sample_dataset["label"].values  # sample feature vector
    target_labels = np.array(
        [get_target_labels(y, true_labels) for y in y_query])

    # Load the serialized transformations previously computed
    logger.info("==> Loading serialized transformations from `{}`".format(
        options['transformations_filename']))
    transformations = load_transformations(options['transformations_filename'])

    spatial_index = None
    if options['index_filename']:
        spatial_index = load_spatial_index(options['index_filename'])

    start_time = time.time()
    knn = get_knn(X_query, y_query, target_labels, sample_ids,
                  transformations, k=options['knn'], spatial_index=spatial_index)
    end_time = int(time.time() - start_time)
    logger.info("Total elapsed time for computing k-NN [k={}]: {:02d}:{:02d}:{:02d}".format(
        options['knn'], end_time // 3600, (end_time % 3600 // 60), end_time % 60))


if __name__ == '__main__':
    sys.exit(main(get_options()))
