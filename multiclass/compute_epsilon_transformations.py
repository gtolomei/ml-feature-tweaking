#!/usr/bin/env python3
# encoding: utf-8

"""
compute_epsilon_transformations.py

Created by Gabriele Tolomei on 2019-04-10.
"""

import sys
import os
import argparse
import logging
import pickle
import ast
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
        'output_dirname',
        help="""Path to the output directory containing results.""",
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
    options['output_dirname'] = args.output_dirname
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


def loading_dataset(input_filename, sep="\t", fillna=None):
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

##########################################################################


def loading_model(model_filename):
    """
    This function loads a model from a dump done via scikit-learn

    Args:
        model_filename (str): path to the file where the model has been serialized and persisted

    Return:
        an instance representing the trained model
    """
    return joblib.load(model_filename)

##########################################################################

##########################################################################


def loading_paths(paths_filename):
    """
    This function return the internal representation of (positive) paths as extracted from
    the model

    Args:
        paths_filename (str): path to the filename containing the persisted positive paths

    Return:
        paths (dict): a dictionary containing a key for each decision tree of the ensemble
                        and for each key a list of tuple (x_i, theta_i) encoding the boolean condition
                        as follows:
                        if theta_i < 0 then the encoded condition will be feature x_i <= theta_i
                        if theta_i > 0 then the encoded condition will be feature x_i > theta_i
    """

    with open(paths_filename, "rb") as paths_file:
        paths = pickle.load(paths_file)

    # paths = {}
    # with open(paths_filename) as paths_file:
    #     for record in paths_file:
    #         record = record.strip()
    #         record = record[1:-2]
    #         # add an extra comma at the end to be able to deal with a
    #         # single-condition path
    #         record = record + ','
    #         fields = record.split(", [")
    #         tree_id = int(fields[0])
    #         path = list(ast.literal_eval(fields[1]))
    #         if tree_id in paths:
    #             paths[tree_id].append(path)
    #         else:
    #             paths[tree_id] = [path]
    return paths

##########################################################################

##########################################################################

# Compute epsilon-transformation of an instance according to a specific
# path ##


def compute_epsilon_transformation(x, path, epsilon, cache):
    """
    This function computes the epsilon-transformation of an original instance x, associated with
    the boolean conditions encoded in the specified path

    Args:
        x (pandas.Series or numpy.ndarray): the original instance x = (x_0, x_1, ..., x_{n-1})
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

    Returns:
        x_prime (numpy.ndarray): a synthetic instance x_prime from the original x, such that it satisfies 
                                the conditions encoded in path with an epsilon tolerance
                                For example, if x = (1.2, -3.7, 0.8) and path = [(0, <=, 1.5), (1, <=, -4)]
                                x_prime = (1.2, -4-epsilon, 0.8)
                                Indeed, the first boolean condition encoded in the path states that
                                - (x_{0+1} <= 1.5) = (x_1 <= 1.5) Since x_1 = 1.2 this condition is already satisfied
                                - (x_{1+1} <= -4) = (x_2 <= -4) Since x_2 = -3.7 this value must be changed accordingly
                                so to satisfy the path, namely we set x_2 = -4-epsilon
                                - Finally, since there is no condition for x_3, we let it as it is.
    """

    logger = logging.getLogger(__name__)

    # # Copy the original input vector using pandas.Series.copy() method
    x_prime = x.copy()

    logger.debug("Loop through all the conditions encoded in the path")
    i = 1

    for cond in path:
        feature = cond[0]  # feature id
        direction = cond[1]  # condition direction (i.e. "<=" or ">")
        threshold = cond[2]  # condition threshold

        # 1. if we already examined this condition for this instance x then
        # we just retrieve the correct feature value for the transformed
        # instance x'
        logger.debug("Check if path condition n. {} = [(x_{}, {}, {})] has been already examined for this instance x...".format(
            i, feature, direction, threshold))
        if cond in cache:
            logger.debug("!!!!! CACHE HIT !!!!!! Path condition n. {} = [(x_{}, {}, {})] has been already examined for this instance x! Let's change x_{} = {} to {:.5f}".format(
                i, feature, direction, threshold, feature, x[feature], cache[cond]))
            x_prime[feature] = cache[cond]
        # 2. otherwise, we must compute the new feature value for the
        # transformed instance x'
        else:
            logger.debug("Path condition n. {} = [(x_{}, {}, {})] has not been yet examined: Let's compute it!".format(
                i, feature, direction, threshold))

            # Negative Direction Case: (x_i, theta_i, <=) ==> x_i must be less than or equal
            # to theta_i (x_i <= theta_i)
            if direction == "<=":
                logger.debug("Direction is \"{}\"".format(direction))
                logger.debug("Condition n. {} is about feature x_{} = {}: ({} {} {})?".format(
                    i, feature, x[feature], x[feature], direction, threshold))
                if x[feature] <= threshold:
                    logger.debug("Condition n. {} is already verified by x as x_{} = {} {} {}".format(
                        i, feature, x[feature], direction, threshold))
                else:
                    logger.debug("Condition n. {} is broken by x as x_{} = {} > {}".format(
                        i, feature, x[feature], threshold))
                    logger.debug("Let x_{} = ({} - epsilon) = ({} - {}) = {}".format(
                        feature, threshold, threshold, epsilon, (threshold - epsilon)))
                    x_prime[feature] = threshold - epsilon

            # Positive Direction Case: (x_i, theta_i, >) ==> x_i must be greater than
            # theta_i (x_i > theta_i)
            else:
                logger.debug("Direction is \"{}\"".format(direction))
                logger.debug("Condition n. {} is about feature x_{} = {}: ({} {} {})?".format(
                    i, feature, x[feature], x[feature], direction, threshold))
                if x[feature] > threshold:
                    logger.debug("Condition n. {} is already verified by x as x_{} = {} {} {}".format(
                        i, feature, x[feature], direction, threshold))
                else:
                    logger.debug("Condition n. {} is broken by x as x_{} = {} <= {}".format(
                        i, feature, x[feature], threshold))
                    logger.debug("Let x_{} = ({} + epsilon) = ({} + {}) = {}".format(
                        feature, threshold, threshold, epsilon, (threshold + epsilon)))
                    x_prime[feature] = threshold + epsilon

            logger.debug("Eventually, let's store feature x_{} = {} just computed according to path condition n. {}".format(
                feature, x_prime[feature], i))
            cache[cond] = x_prime[feature]

        i += 1

    return x_prime

##########################################################################


def compute_epsilon_transformations(x, i, model, k, paths, epsilon):

    logger = logging.getLogger(__name__)

    candidate_transformations = []
    cache = {}
    # Loop through all the trees of the ensemble
    for tree_id, tree in enumerate(model.estimators_):
        logger.info("Examining tree ID #{}".format(tree_id))
        logger.info(
            "Retrieve all the {}-leaved paths from tree ID #{}".format(k, tree_id))
        k_leaved_paths = paths[tree_id]
        # Loop through all the k-leaved paths of this tree
        for path_id, path in enumerate(k_leaved_paths):
            logger.debug(
                "Compute the {}-labelled epsilon-transformation of instance ID #{} from path ID #{} of tree ID #{}".format(k, i, path_id, tree_id))
            x_prime = compute_epsilon_transformation(
                x, path, epsilon, cache)
            logger.debug(
                "Check if the {}-labelled epsilon-transformation of instance ID #{} just computed is also a candidate transformation".format(k, i))
            if model.predict(x_prime.reshape(1, -1))[0] == k:
                logger.info(
                    "Add the {}-labelled epsilon-transformation of instance ID #{} from path ID #{} of tree ID #{} to the list of candidates".format(k, i, path_id, tree_id))
                candidate_transformations.append(x_prime)

    return candidate_transformations

##########################################################################

# Map function to compute epsilon-transformations of an instance to a target class label k


def map_compute_epsilon_transformations(instance):

    logger = logging.getLogger(__name__)

    x, i, model, label, paths, epsilon = instance

    logger.info(
        "Computing all the possible {}-labelled epsilon-transformations for instance x ID #{}".format(label, i))

    return ((label, i), compute_epsilon_transformations(x, i, model, label, paths, epsilon))

##########################################################################


def extract_correctly_predicted_instances(dataset, model):

    X = dataset.iloc[:, 1:].values
    y = dataset["label"].values

    return dataset[(model.predict(X) == y)]

##########################################################################


def main(options):
    logger = configure_logging(level=logging.INFO)

    # Loading dataset
    logger.info("==> Loading dataset from `{}`".format(
        options['dataset_filename']))
    dataset = loading_dataset(options['dataset_filename'], sep=",")

    logger.info("Shape of the dataset: {} instances by {} features".format(
        dataset.shape[0], dataset.shape[1] - 1))

    # Loading model
    logger.info("==> Loading model from `{}`".format(
        options['model_filename']))
    model = loading_model(options['model_filename'])

    # Loading all the paths encoded by the model
    logger.info("==> Loading paths from `{}`".format(
        options['paths_filename']))
    paths = loading_paths(options['paths_filename'])

    # Working only on those instances which the model is able to correctly predict the true class of
    logger.info("==> Extracting from the original dataset those instances which the model is able to correctly predict the true class of")
    dataset = extract_correctly_predicted_instances(dataset, model)

    logger.info("==> Creating the pool of {} workers".format(mp.cpu_count()))
    pool = mp.Pool(mp.cpu_count())
    logger.info("==> Preparing the input to be sent to each worker of the pool")

    dataset = dataset.iloc[:1, :]

    k_labelled_transformations = {}

    for label in [7]:  # model.classes_:
        logger.info(
            "Transform all non-{}-labelled instances into {}-labelled ones".format(label, label))
        X = dataset[~(dataset["label"] == label)].iloc[:, 1:]
        logger.info(
            "Number of instances to be transformed: {}".format(X.shape[0]))
        inputs = [(X.loc[i, :].values, i, model, label,
                   paths[label], options['epsilon']) for i in X.index]
        result = pool.map(map_compute_epsilon_transformations, inputs)
        for k, v in result:
            k_labelled_transformations.setdefault(k, []).append(v)

    print(k_labelled_transformations)

    # # Compute all the candidate epsilon-transformations of all the instances from their original label to any other target label
    # logger.info(
    #     "==> Compute all the candidate {}-transformations of all the instances from their original label to any other target label".format(options['epsilon']))
    # candidate_transformations = compute_candidate_transformations(
    #     dataset.head(100), model, paths, epsilon=options['epsilon'])


if __name__ == '__main__':
    sys.exit(main(get_options()))
