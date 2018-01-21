#!/usr/bin/env python
# encoding: utf-8
"""
tweak_features.py
"""

import sys
import os
import argparse
import logging
import logging.handlers
import ast
import pandas as pd
import multiprocessing as mp
from sklearn.externals import joblib

# console logging format
CONSOLE_LOGGING_FORMAT = '%(asctime)-15s *** %(levelname)s *** %(message)s'
# file logging format
FILE_LOGGING_FORMAT = '%(asctime)-15s *** %(levelname)s [%(filename)s:%(lineno)s - %(funcName)s()] *** %(message)s'

# get the root logger
logger = logging.getLogger(__name__)
# set the logging level (default: DEBUG)
logger.setLevel(logging.DEBUG)
# create a stream handler associated with the console (stdout)
console_handler = logging.StreamHandler(sys.stdout)
# set the console handler logging format
console_logging_format = logging.Formatter(CONSOLE_LOGGING_FORMAT)
# specify the logging format for this console handler
console_handler.setFormatter(console_logging_format)
# set the logging level for this console handler (default: INFO)
console_handler.setLevel(logging.INFO)
# attach this console handler to the logger
logger.addHandler(console_handler)
# create a rotating file handler associated with an external file
file_handler = logging.handlers.RotatingFileHandler(
    "tweak_features_" + str(os.getpid()) + ".log", mode='w', maxBytes=(1048576 * 5), backupCount=2, encoding=None, delay=0)
# set the file handler logging format
file_logging_format = logging.Formatter(FILE_LOGGING_FORMAT)
# specify the logging format for this file handler
file_handler.setFormatter(file_logging_format)
# set the logging level for this file handler (default: DEBUG)
file_handler.setLevel(logging.DEBUG)
# attach this file handler to the logger
logger.addHandler(file_handler)


def get_options(cmd_args=None):
    """
    Parse command line arguments
    """
    cmd_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmd_parser.add_argument(
        'dataset_filename',
        help="""Path to the input file containing the original dataset (features + class labels).""",
        type=str)
    cmd_parser.add_argument(
        'model_filename',
        help="""Path to the input file containing the serialized model.""",
        type=str)
    cmd_parser.add_argument(
        'paths_filename',
        help="""Path to the input file containing the (positive) paths represented by the model.""",
        type=str)
    cmd_parser.add_argument(
        'output_dirname',
        help="""Path to the output directory containing results.""",
        type=str)
    cmd_parser.add_argument(
        '-e',
        '--epsilon',
        help="""Tolerance used to pass the boolean tests encoded in each decision tree.""",
        type=check_valid_epsilon,
        default=0.01)

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
            "%s is an invalid value for test_split_proportion which must be any x, such that x > 0" % fvalue)
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
    paths = {}
    with open(paths_filename) as paths_file:
        for record in paths_file:
            record = record.strip()
            record = record[1:-2]
            # add an extra comma at the end to be able to deal with a
            # single-condition path
            record = record + ','
            fields = record.split(", [")
            tree_id = int(fields[0])
            path = list(ast.literal_eval(fields[1]))
            if tree_id in paths:
                paths[tree_id].append(path)
            else:
                paths[tree_id] = [path]
    return paths

##########################################################################

############################ Retrieve True Negative Instances ############


def get_true_negatives(X, y, model):
    """
    This function retrieves those instances whose class labels are really negative
    and which also the model correctly predict as negative

    Args:
        X (pandas.DataFrame): the matrix of features (m x n)
        y (pandas.Series): the vector of class labels (m x 1)
        model (sklearn.ensemble): the trained (ensemble) classifier

    Returns:
        true_negatives (numpy.array): the list of record x_i in X whose class label y_i = -1
                                      and whose predicted class label y_i_hat = -1 as well
    """
    true_negatives = []
    for i in X.index:
        logger.info("Prediction for instance ID#%d" % i)
        x_i = X.ix[i]
        y_i = y.ix[i]
        # DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and willraise ValueError in 0.19.
        # Reshape your data either using X.reshape(-1, 1) if your data has a single feature or
        # X.reshape(1, -1) if it contains a single sample.
        y_i_hat = model.predict(x_i.reshape(1, -1))[0]
        logger.info(
            "True Class Label = %d; Predicted Class Label = %d" % (y_i, y_i_hat))
        if y_i == y_i_hat:
            logger.info(
                "True and Predicted Class Labels are negative! Let's add the corresponding index to the final list")
            true_negatives.append(i)
    return true_negatives


##########################################################################

# Map function to compute epsilon-transformation of an instance

def map_compute_epsilon_transformation(instance):
    x, n, i, epsilon, model, paths = instance

    logger.info(
        "Computing all the possible epsilon-transformations for instance x n. %d [ID#%d]" % (n, i))
    return (i, compute_epsilon_transformations_of_instance(x, n, i, epsilon, model, paths))


##########################################################################

############### Compute epsilon-transformation of an instance ############

def compute_epsilon_transformations_of_instance(x, n, i, epsilon, model, paths):

    x_transformations = {}
    path_conditions = {}
    tree_id = 0
    logger.info("Loop through all the decision trees of the ensemble...")
    for decision_tree in model.estimators_:
        logger.debug(
            "Check if the prediction for the instance x n. %d [ID#%d] of the base decision tree ID #%d coincides with the overall prediction of the ensemble" % (n, i, tree_id))
        y_hat_dt = model.classes_[
            int(decision_tree.predict(x.reshape(1, -1))[0])]
        logger.debug("Class Label Prediction for x n. %d [ID#%d] according to the base decision tree ID #%d = %d" % (
            n, i, tree_id, y_hat_dt))
        y_hat_ensemble = model.predict(x.reshape(1, -1))
        logger.debug("Class Label Prediction for x n. %d [ID#%d] according to overall ensemble = %d" % (
            n, i, y_hat_ensemble))

        if y_hat_dt == y_hat_ensemble:
            logger.debug(
                "Both class label predictions are the same (and they are both negatives)")

            logger.debug(
                "Get all the positive paths of the decision tree ID #%d" % tree_id)
            paths_dt = paths[tree_id]
            logger.info(
                "Loop through all the positive paths of the decision tree ID #%d" % tree_id)
            path_id = 0
            for path in paths_dt:
                logger.debug(
                    "Compute x' as the epsilon-transformation of x n. %d [ID#%d] using the positive path ID #%d (length=%d) of tree ID #%d" % (n, i, path_id, len(path), tree_id))
                x_prime = compute_epsilon_transformation_path(
                    x, epsilon, path, path_conditions)

                x_prime_class = model.classes_[
                    int(decision_tree.predict(x_prime.reshape(1, -1))[0])]

                logger.debug(
                    "The predicted class label by tree ID #%d for this epsilon-transformation x' according to path ID #%d is %d" % (tree_id,
                                                                                                                                    path_id, x_prime_class))

                if x_prime_class != 1:
                    logger.warn(
                        "The predicted class label by tree ID #%d for this epsilon-transformation x' according to path ID #%d should be 1" % (tree_id, path_id))

                else:
                    logger.debug(
                        "Check if this epsilon-transformation leads to an overall positive prediction...")
                    logger.debug(
                        "The overall predicted class label by the ensemble for this epsilon-transformation x' is %d" % (model.predict(x_prime.reshape(1, -1))[0]))

                    if model.predict(x_prime.reshape(1, -1))[0] == 1:
                        logger.info(
                            "This epsilon-transformation of x n. %d [ID#%d] leads to an overall positive prediction and therefore is a candidate transformation for x" % (n, i))

                        candidate = (path_id, len(path), x_prime)
                        if tree_id in x_transformations:
                            x_transformations[tree_id].append(candidate)
                        else:
                            x_transformations[tree_id] = [candidate]
                path_id += 1

        tree_id += 1

    return x_transformations

##########################################################################

# Compute epsilon-transformation of an instance according to a specific
# path ##


def compute_epsilon_transformation_path(x, epsilon, path, path_conditions):
    """
    This function computes the epsilon transformation of an instance x
    according to the boolean conditions encoded in the specified path

    Args:
        x (pandas.Series): vector representing the instance
                                                x = (x_1, x_2, ..., x_n)
        epsilon (float): tolerance used to pass the tests encoded in path
        path (list(tuple)): encoding of a root-to-leaf path of a decision tree as
                            [(0, <dir>, theta_0), ..., (n-1, <dir>, theta_{n-1})] 
                            where each (i, <dir>, theta_i) encode a boolean condition as follows
                            - if <dir> = "<=" then (i, "<=", theta_i) means that the (i+1)-th feature must be less than or equal to theta_i
                            (x_{i+1} <= theta_i)
                            - if <dir> = ">" then (i, ">", theta_i) means that the (i+1)-th feature must be greater than theta_i
                            (x_{i+1} > theta_i)
                            (Note: the discrepancy of the indices derives from the fact that features are 0-based indexed on the path,
                            although usually they are referred using 1-based notation)

    Returns:
        tuple(x_prime, cost) where
            x_prime (pandas.Series): a transformation of the original
                                                        instance x so that x_prime satisfies
                                                        the conditions encoded in path with an epsilon tolerance
                                                        For example, if x = (1.2, -3.7, 0.8) and path = [(0, <=, 1.5), (1, <=, -4)]
                                                        x_prime = (
                                                            1.2, -4-epsilon, 0.8)
                                                        Indeed, the first boolean condition encoded in the path states that
                                                        - (x_{0+1} <= 1.5) = (x_1 <= 1.5) Since x_1 = 1.2 this condition is already satisfied
                                                        - (x_{1+1} <= -4) = (x_2 <= -4) Since x_2 = -3.7 this value must be changed accordingly
                                                        so to satisfy the path, namely we set x_2 = -4-epsilon
                                                        - Finally, since there is no condition for x_3, we let it as it is.
    """

    # Copy the original input vector using pandas.Series.copy() method
    x_prime = x.copy()

    logger.info("Loop through all the conditions encoded in the path")
    i = 1

    for cond in path:
        feature = cond[0]  # feature id
        direction = cond[1]  # condition direction (i.e. "<=" or ">")
        threshold = cond[2]  # condition threshold

        # 1. if we already examined this condition for this instance x then
        # we just retrieve the correct feature value for the transformed
        # instance x'
        logger.info("Check if path condition n. %d = [(%d, %s, %.5f)] has been already examined for this instance x..." % (
            i, feature, direction, threshold))
        if cond in path_conditions:
            logger.info("Path condition n. %d = [(%d, %s, %.5f)] has been already examined for this instance x! Let's assign %s = %.5f"
                        % (i, feature, direction, threshold, x.index[feature], path_conditions[cond]))
            x_prime[feature] = path_conditions[cond]
        # 2. otherwise, we must compute the new feature value for the
        # transformed instance x'
        else:
            logger.info("Path condition n. %d = [(%d, %s, %.5f)] has not been yet examined: Let's compute it!" % (
                i, feature, direction, threshold))

            # Negative Direction Case: (x_i, theta_i, <=) ==> x_i must be less than or equal
            # to theta_i (x_i <= theta_i)
            if direction == "<=":
                logger.debug("Direction is \"%s\"" % direction)
                logger.debug("Condition n. %d is about feature x_%d = %s: [(%s %s %.5f)]" % (
                    i, feature + 1, x.index[feature], x.index[feature], direction, threshold))
                if x[feature] <= threshold:
                    logger.debug("Condition n. %d is already verified by x as %s = %.5f" % (
                        i, x.index[feature], x[feature]))
                else:
                    logger.debug("Condition n. %d is broken by x as %s = %.5f" % (
                        i, x.index[feature], x[feature]))
                    logger.debug("Let %s = (%.5f - epsilon) = (%.5f - %.5f) = %.5f" %
                                 (x.index[feature], threshold, threshold, epsilon, (threshold - epsilon)))
                    x_prime[feature] = threshold - epsilon

            # Positive Direction Case: (x_i, theta_i, >) ==> x_i must be greater than
            # theta_i (x_i > theta_i)
            else:
                logger.debug("Direction is \"%s\"" % direction)
                logger.debug("Condition n. %d is about feature x_%d = %s: [(%s %s %.5f)]" % (
                    i, feature + 1, x.index[feature], x.index[feature], direction, threshold))
                if x[feature] > threshold:
                    logger.debug("Condition n. %d is already verified by x as %s = %.5f" % (
                        i, x.index[feature], x[feature]))
                else:
                    logger.debug("Condition n. %d is broken by x as %s = %.5f" % (
                        i, x.index[feature], x[feature]))
                    logger.debug("Let %s = (%.5f + epsilon) = (%.5f + %.5f) = %.5f" %
                                 (x.index[feature], threshold, threshold, epsilon, (threshold + epsilon)))
                    x_prime[feature] = threshold + epsilon

            logger.info("Eventually, let's store feature %s = %.5f just computed according to path condition n. %d"
                        % (x.index[feature], x_prime[feature], i))
            path_conditions[cond] = x_prime[feature]

        i += 1

    return x_prime


##########################################################################

################# Store epsilon-transformations ##########################

def save_epsilon_transformations(X_transformations, outfilename, sep="\t", header=None):
    with open(outfilename, 'w') as outfile:
        out_fmt = "%d" + sep + "%d" + sep + "%d" + sep + "%d" + sep + "%s\n"
        if header:
            outfile.write(sep.join(header) + "\n")
        for key in X_transformations:
            for tree_id in sorted(X_transformations[key]):
                for element in X_transformations[key][tree_id]:
                    path_id = element[0]
                    path_length = element[1]
                    x_prime = element[2]
                    outfile.write(out_fmt % (
                        key, tree_id, path_id, path_length, sep.join([str(x) for x in x_prime])))


##########################################################################

############################## Main ######################################

def main(options):

    logger.info("Loading dataset from " + options['dataset_filename'])
    # Loading dataset
    dataset = loading_dataset(options['dataset_filename'])

    logger.info("Loading model from " + options['model_filename'])
    # Loading model
    model = loading_model(options['model_filename'])

    logger.info("Loading positive paths from " + options['paths_filename'])
    # Loading the positive paths of the model
    paths = loading_paths(options['paths_filename'])

    logger.info(
        "Selecting only (true) negative instances from the portion of dataset indexed")
    dataset = dataset[dataset["class"] == -1]

    logger.info(
        "Selecting features of only (true) negative instances from the portion of dataset indexed")
    # Features
    X = dataset.iloc[:, :len(dataset.columns) - 1]
    logger.info(
        "Selecting class labels of only (true) negative instances from the portion of dataset indexed")

    # Class labels
    y = dataset.iloc[:, len(dataset.columns) - 1]

    logger.info(
        "Retrieving the list of instances whose class labels are really negatives and that are correctly predicted as negatives")
    true_negatives = get_true_negatives(X, y, model)

    X_negatives = X.ix[true_negatives]

    logger.info("Creating the pool of workers")
    pool = mp.Pool()
    logger.info("Preparing the input to be sent to each worker of the pool")
    X_inputs = zip(range(0, len(X_negatives.index)), X_negatives.index)  # idx
    inputs = [(X_negatives.ix[i], n, i, options['epsilon'], model, paths)
              for (n, i) in X_inputs]

    logger.info("Compute all the possible epsilon-transformations in parallel for all the true negative instances of the dataset using workers of the pool")
    X_negatives_transformations = dict(
        pool.map(map_compute_epsilon_transformation, inputs))

    logger.info("Creating the header for the output file")
    header = ['id', 'tree_id', 'path_id',
              'path_length'] + list(X.columns.values)

    # Save epsilon-transformations to disk
    logger.info("Save the computed epsilon-transformations to %s" %
                options['output_dirname'] + '/transformations_' + str(options['epsilon']) + '.tsv')
    save_epsilon_transformations(X_negatives_transformations, options[
        'output_dirname'] + '/transformations_' + str(options['epsilon']) + '.tsv', header=header)


if __name__ == "__main__":
    sys.exit(main(get_options()))
