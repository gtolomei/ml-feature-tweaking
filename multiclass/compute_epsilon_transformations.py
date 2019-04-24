#!/usr/bin/env python3
# encoding: utf-8

"""
compute_epsilon_transformations.py

Created by Gabriele Tolomei on 2019-04-10.
"""

import sys
import argparse
import logging
import gzip
import numpy as np
import pandas as pd
import multiprocessing as mp

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
        filename="./compute_epsilon_transformations.log", mode="w")
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
        'model_filename',
        help="""Path to the file containing the serialized ensemble of decision trees (i.e., an instance of BaseEstimator class).""",
        type=str)
    cmd_parser.add_argument(
        'paths_filename',
        help="""Path to the input file containing all the paths represented by the model.""",
        type=str)
    cmd_parser.add_argument(
        'output_filename',
        help="""Path to the output file containing all the k-labelled transformations.""",
        type=str)
    cmd_parser.add_argument(
        '-e',
        '--epsilon',
        help="""Tolerance used to pass the boolean tests encoded by each decision tree.""",
        type=check_valid_epsilon,
        default=1)
    args = cmd_parser.parse_args(cmd_args)

    options = {}
    options['dataset_filename'] = args.dataset_filename
    options['model_filename'] = args.model_filename
    options['paths_filename'] = args.paths_filename
    options['output_filename'] = args.output_filename
    options['epsilon'] = args.epsilon

    return options

######################## Check Input Validity ###################


def check_valid_epsilon(value):
    """
    This function is responsible for checking the validity of the input threshold epsilon.

    Args:
        value (str): value passed as input argument to this script

    Return:
        a float if value is such that value > 0, an argparse.ArgumentTypeError otherwise
    """
    fvalue = float(value)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError(
            "{} is an invalid value for test_split_proportion which must be any x, such that x > 0".format(fvalue))
    return fvalue

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
    return joblib.load(model_filename)

##########################################################################


def load_paths(paths_filename):
    """
    This function return the internal representation of paths as extracted from
    the learned model

    Args:
        paths_filename (str): path to the filename containing the persisted positive paths

    Return:
        paths (dict):   a dictionary of dictionaries.
                        The outermost dictionary contains a key for each class label.
                        Each innermost dictionary contains a key for each decision tree of the ensemble.
                        Each entry is in turn made of a list of list of tuples, where each tuple is in the form of (x_i, dir, theta_i)
                        encoding the boolean condition as follows:
                            x_i <= theta_i, if dir = "<="
                            x_i > theta_i, if dir = ">"
                        e.g., paths[k][tree_id] = [[(14, <=, -0.7171), (7, >, 457.0), (12, <=, 54.609), (39, >, -0.059)], ...]
    """

    with open(paths_filename, "rb") as paths_file:
        paths = joblib.load(paths_file)

    return paths

##########################################################################


def compute_k_labelled_instance(path, epsilon, size, dtype, cache):
    """
    This function computes the epsilon-transformation of an original instance x, associated with
    the boolean conditions encoded in the specified path

    Args:
        path (list(tuple)): encoding of a root-to-leaf path of a decision tree as
                            [(0, <dir>, theta_0), ..., (n-1, <dir>, theta_{n-1})]
                            where each (i, <dir>, theta_i) encodes a boolean condition as follows
                            - if <dir> = "<=" then (i, "<=", theta_i) means that the (i+1)-th feature must be less than or equal to theta_i
                            (x_{i+1} <= theta_i)
                            - if <dir> = ">" then (i, ">", theta_i) means that the (i+1)-th feature must be greater than theta_i
                            (x_{i+1} > theta_i)
                            (Note: the discrepancy of the indices derives from the fact that features are 0-based indexed on the path,
                            although usually they are referred using 1-based notation)
        epsilon (float): tolerance used to pass the tests encoded in path
        size (int): number of total element (i.e., features) of the (syntetic) k-labelled instance
        dtype (numpy.dtype): dtype associated with the elements of the returned instance

    Returns:
        x_synt (numpy.ndarray): a synthetic instance, such that it satisfies
                                the conditions encoded in path with an epsilon tolerance
                                For example, if path = [(0, >, 1.5), (2, <=, -4)] and size = 5
                                x_synt = (1.5+epsilon, 0, -4-epsilon, 0, 0)
    """

    logger = logging.getLogger(__name__)

    logger.debug("Loop through all the conditions encoded in the path")
    i = 1

    x_synt = np.zeros(size, dtype=dtype)

    for cond in path:
        feature = cond[0]  # feature id
        direction = cond[1]  # condition direction (i.e. "<=" or ">")
        threshold = cond[2]  # condition threshold

        # 1. if we already examined this condition for this instance x then
        # we just retrieve the correct feature value for the transformed
        # instance x'
        logger.debug("Check if path condition n. {} = [(x_{}, {}, {})] has been already examined".format(
            i, feature, direction, threshold))
        if cond in cache:
            logger.debug("!!!!! CACHE HIT !!!!!! Path condition n. {} = [(x_{}, {}, {})] has been already examined! Let's change x_{} to {:.5f}".format(
                i, feature, direction, threshold, feature, cache[cond]))
            x_synt[feature] = cache[cond]
        # 2. otherwise, we must compute the new feature value for the
        # transformed instance x'
        else:
            logger.debug("Path condition n. {} = [(x_{}, {}, {})] has not been yet examined: Let's compute it!".format(
                i, feature, direction, threshold))

            # Negative Direction Case: (x_i, theta_i, <=) ==> x_i must be less than or equal
            # to theta_i (x_i <= theta_i)
            if direction == "<=":
                logger.debug("Condition n. {} is: feature x_{} {} {}".format(
                    i, feature, direction, threshold))
                logger.debug("Let x_{} = ({} - epsilon) = ({} - {}) = {}".format(
                    feature, threshold, threshold, epsilon, (threshold - epsilon)))
                x_synt[feature] = threshold - epsilon

            # Positive Direction Case: (x_i, theta_i, >) ==> x_i must be greater than
            # theta_i (x_i > theta_i)
            else:
                logger.debug("Condition n. {} is: feature x_{} {} {}".format(
                    i, feature, direction, threshold))
                logger.debug("Let x_{} = ({} + epsilon) = ({} + {}) = {}".format(
                    feature, threshold, threshold, epsilon, (threshold + epsilon)))
                x_synt[feature] = threshold + epsilon

            logger.debug("Eventually, let's store feature x_{} = {} just computed according to path condition n. {}".format(
                feature, x_synt[feature], i))
            cache[cond] = x_synt[feature]

        i += 1

    return x_synt


##########################################################################


def create_synthetic_instance(x, path):
    x_synt = path.copy()
    # mask = path == 0
    # x_synt[mask] = path[mask] + x[mask]

    return x_synt

##########################################################################


def compute_k_labelled_instances(X, model, paths, k, epsilon, size, dtype=int):

    logger = logging.getLogger(__name__)

    X_candidates = []  # np.array([], dtype=dtype).reshape((0, size))

    cache = {}
    # Loop through all the trees of the ensemble
    for tree_id, tree in enumerate(model.estimators_[:5]):
        logger.info("Examining tree ID #{}".format(tree_id))
        logger.info(
            "Retrieve all the {}-leaved paths from tree ID #{}".format(k, tree_id))
        k_leaved_paths = paths[tree_id]

        # Loop through all the k-leaved paths of this tree
        for path_id, path in enumerate(k_leaved_paths):
            logger.debug(
                "Compute the {}-labelled epsilon-bound transformation (i.e., synthetic instance) from path ID #{} of tree ID #{}".format(k, path_id, tree_id))
            x_path = compute_k_labelled_instance(
                path, epsilon, size, dtype, cache)
            # Smartly combine the computed synthetic instance derived from this k-leaved path with all the non-k-labelled instances
            logger.debug(
                "Create the corresponding synthetic instances combining all the {} non-{}-labelled instances".format(X.shape[0], k))
            X_synt = np.apply_along_axis(
                create_synthetic_instance, 1, X, x_path)
            # Restrict synthetic instances to those actually leading to a k-labelled prediction
            # using boolean mask indexing which synthetic instance has actually switched prediction (i.e., from non-k to k)
            logger.debug(
                "Restrict to only those synthetic instances which actually switch their prediction to {}".format(k))
            X_synt_candidates = X_synt[model.predict(X_synt) == k]
            # Concatenate the computed synthetic candidate instances to the list of all candidates
            logger.debug(
                "Add these synthetic instances to the final list of candidates")
            X_candidates.extend(X_synt_candidates)
            # X_candidates = np.concatenate(
            #     (X_candidates, X_synt_candidates), axis=0)

    logger.info("Eventually, return all the candidate {}-bound {}-labelled transformations [n. of candidates = {}]".format(
        epsilon, k, len(X_candidates)))

    return X_candidates

##########################################################################


def map_compute_epsilon_transformations(instance):

    logger = logging.getLogger(__name__)

    X, model, paths, label, epsilon, size, dtype = instance

    logger.info(
        "Computing all the possible {}-labelled epsilon-transformations".format(label))

    return ((label), compute_k_labelled_instances(X, model, paths, label, epsilon, size, dtype=dtype))

##########################################################################


def extract_correctly_predicted_instances(dataset, model, label="label"):

    X = dataset.iloc[:, 1:].values
    y = dataset[label].values

    return dataset[(model.predict(X) == y)]

##########################################################################


def save_transformations(transformations, output_filename):
    with gzip.GzipFile(output_filename + '.gz', 'wb') as output_file:
        joblib.dump(transformations, output_file)


##########################################################################


def main(options):

    logger = configure_logging(level=logging.INFO)

    # Loading dataset
    logger.info("==> Loading dataset from `{}`".format(
        options['dataset_filename']))
    dataset = load_dataset(options['dataset_filename'], sep=",")

    logger.info("Shape of the dataset: {} instances by {} features".format(
        dataset.shape[0], dataset.shape[1] - 1))

    # Loading model
    logger.info("==> Loading model from `{}`".format(
        options['model_filename']))
    model = load_model(options['model_filename'])

    # Loading all the paths encoded by the model
    logger.info("==> Loading paths from `{}`".format(
        options['paths_filename']))
    paths = load_paths(options['paths_filename'])

    # Working only on those instances which the model is able to correctly predict the true class of
    logger.info("==> Extracting from the original dataset those instances which the model is able to correctly predict the true class of")
    dataset = extract_correctly_predicted_instances(dataset, model)

    # Setting up multiple processes (i.e., workers)
    logger.info("==> Creating the pool of {} workers".format(mp.cpu_count()))
    pool = mp.Pool(mp.cpu_count())
    logger.info("==> Preparing the input to be sent to each worker of the pool")

    # Compute all the candidate epsilon-transformations of all the instances from their original label to any other target label
    logger.info(
        "==> Compute all the candidate {}-transformations of all the instances from their original label to any other target label".format(options['epsilon']))

    dataset = dataset.iloc[:1, :]

    # dictionary of all k-labelled transformations {'label': [trans_1, trans_2, ..., trans_n]}
    # where each `trans_i` is a one-dimensional numpy array
    k_labelled_transformations = {}
    # list of inputs sent to each worker [(input_w1), ..., (input_wm)]
    inputs = []
    # loop through every label
    for label in model.classes_:
        logger.info(
            "Transform all non-{}-labelled instances into {}-labelled ones".format(label, label))
        # select only the subset of instances having a different label from the one currently under investigation
        X = dataset[~(dataset["label"] == label)].iloc[:, 1:].values
        # check if there exists at least one instance having such a property
        if X.shape[0] > 0:
            logger.info(
                "Number of instances to be transformed to label k = {}: {}".format(label, X.shape[0]))
            # if so, just append the portion of the dataset (plus extra arguments) to the list of inputs
            inputs.append((X, model, paths[label], label,
                           options['epsilon'], X.shape[1], X.dtype))
        else:
            logger.info(
                "All the instances have already label k = {}".format(label))

    # the output of all the workers will be a tuple of tuples ('label', [trans_1, trans_2, ..., trans_n])
    # by applying a `dict` operator this will be transformed into a dictionary as expected
    k_labelled_transformations = dict(
        pool.map(map_compute_epsilon_transformations, inputs))

    # Finally, persist the just computed k-labelled transformations to disk
    logger.info("Finally, serialize transformations to `{}.gz`".format(
        options['output_filename']))
    save_transformations(k_labelled_transformations,
                         options['output_filename'])

    # for k in k_labelled_transformations:
    #     X_trans = k_labelled_transformations[k]
    #     logger.info("Class label: {}".format(k))
    #     if len(X_trans) > 0:
    #         logger.info("Do all transformations lead to a {}-labelled instance? {}".format(
    #             k, np.all(model.predict(X_trans) == k)))
    #     else:
    #         logger.info(
    #             "No transformations available for class label {}".format(k))


if __name__ == '__main__':
    sys.exit(main(get_options()))
