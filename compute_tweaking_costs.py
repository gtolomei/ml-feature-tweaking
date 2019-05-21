#!/usr/bin/env python
# encoding: utf-8
"""
compute_tweaking_costs.py
"""

import sys
import os
import argparse
import logging
import logging.handlers
import inspect
import collections
import cost_functions
import pandas as pd
import numpy as np
from scipy.stats import *

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
    "compute_tweaking_costs_" + str(os.getpid()) + ".log", mode='w', maxBytes=(1048576 * 5), backupCount=2, encoding=None, delay=0)
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
        'tweakings_filename',
        help="""Path to the input file containing the tweaked features.""",
        type=str)
    cmd_parser.add_argument(
        'output_dirname',
        help="""Path to the output directory containing results.""",
        type=str)
    cmd_parser.add_argument(
        '-c',
        '--costfuncs',
        help="""List of function names used to measure the cost associated with each feature tweaking.""",
        type=check_valid_cost_functions,
        default="euclidean_distance")

    args = cmd_parser.parse_args(cmd_args)

    options = {}
    options['dataset_filename'] = args.dataset_filename
    options['tweakings_filename'] = args.tweakings_filename
    options['output_dirname'] = args.output_dirname
    options['costfuncs'] = args.costfuncs

    return options

#################################################################

######################## Check Input Validity ###################


def check_valid_cost_functions(costfuncs):
    """
    This function is responsible for checking the validity of the input cost functions.

    Args:
        costfuncs (str): comma-separated string representing the list of cost functions passed as input argument to this script

    Return:
        a list of strings representing those cost function names specified as input that are also contained in the cost_functions module imported or
        an argparse.ArgumentTypeError otherwise
    """

    name_func_tuples = inspect.getmembers(cost_functions, inspect.isfunction)
    name_func_tuples = [t for t in name_func_tuples if inspect.getmodule(
        t[1]) == cost_functions and not t[0].startswith("__")]

    available_cost_functions = collections.OrderedDict.fromkeys(
        [t[0] for t in name_func_tuples])
    input_cost_functions = collections.OrderedDict.fromkeys(
        [str(f) for f in costfuncs.split(",")])
    # input_cost_functions.intersection(available_cost_functions)
    matched_cost_functions = collections.OrderedDict.fromkeys(
        f for f in input_cost_functions if f in available_cost_functions)
    unmatched_cost_functions = collections.OrderedDict.fromkeys(
        f for f in input_cost_functions if f not in available_cost_functions)

    if not matched_cost_functions:
        raise argparse.ArgumentTypeError(
            "No function in the input list [{}] is a valid cost function! Please choose your input list from the following:\n{}".format(", ".join([cf for cf in input_cost_functions]), "\n".join(["* " + f for f in available_cost_functions])))

    if len(matched_cost_functions) < len(input_cost_functions):
        logger.info("The following input functions are not valid cost functions: [{}]".format(
            ", ".join([f for f in unmatched_cost_functions])))
        logger.info("The cost functions we will be using are the following: [{}]".format(
            ", ".join([f for f in matched_cost_functions])))

    return list(matched_cost_functions)


##########################################################################

##########################################################################

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

############### Compute epsilon-transformation of an instance ############


def compute_transformation_cost(x, x_prime, costfunc):
    """
    This function computes the cost associated with the epsilon-transformation
    of the original feature vector x into x_prime

    Args:
        x (numpy.array): the original feature vector
        x_prime (numpy.array): the epsilon-transformed feature vector
        costfunc (str): the string representing the function used to actually compute the cost of transformation
                         (default='euclidean_distance', i.e. euclidean distance between x and x_prime)
                         Other implementations of the cost function are possible,
                         e.g., the number of features which needed to be tweaked

    Return:
        cost (float): the actual cost of transforming x into x_prime using the specified cost function
    """

    return getattr(cost_functions, costfunc)(x, x_prime)


##########################################################################

# Create two DataFrames, one containing the actual costs of each transformation
# and the other indicating the direction of change for each feature of
# each transformation

def create_tweaked_costs_dataframe(X, X_tweaked, costfuncs):

    # build the heder for the cost DataFrame
    costs_header = ['id', 'tree_id', 'path_id', 'path_length'] + costfuncs
    # build the header for the DataFrame containing the signed transformations
    signs_header = ['id'] + X.columns.values.tolist()

    costs_rows = []
    signs_rows = []

    for i in X.index:
        logger.info(
            "Check if there are any positive transformations associated with the true negative instance x [ID#{}]".format(i))

        if i in X_tweaked.index:
            print(i)
            x = X.ix[i]
            print(x)
            logger.debug(
                "Transform the original feature vector x into a numpy.array object")
            x = np.asarray(x)
            # encapsulate the index i into a list to enforce the .ix method to always return a pandas.DataFrame object
            # indeed, if there is a single entry corresponding to the index i then the .ix method would return a pandas.Series object
            # and the invocation of the .iterrows method (see 5 lines below) is not allowed on a pandas.Series object
            # by encapsulating the index i into a list we are guaranteed to
            # always get back a pandas.DataFrame object
            x_primes = X_tweaked.ix[[i]]
            print(x_primes)
            logger.info("There are {} positive transformations associated with the true negative instance x [ID#{}]".format(
                len(x_primes), i))
            for row in x_primes.iterrows():
                # tree_id = int(row[1][0])
                # path_id = int(row[1][1])
                # path_length = int(row[1][2])
                # x_prime = row[1][3:]
                tree_id = int(row[1][1])
                path_id = int(row[1][2])
                path_length = int(row[1][3])
                x_prime = row[1][4:]
                costs_record = [i, tree_id, path_id, path_length]
                signs_record = [i]
                logger.debug(
                    "Transform the epsilon-transformed feature vector x' into a numpy.array object")
                x_prime = np.asarray(x_prime)
                logger.info("Compute all the costs associated with this transformation of instance x [ID#{}] referring to [tree_id={};path_id={};path_length={}]"
                            .format(i, tree_id, path_id, path_length))

                logger.info("Check if the two vectors are the same")
                if np.array_equal(x, x_prime):
                    logger.info(
                        "The two vectors are indeed the same! The cost for this transformation is just 0")
                    for cf in cosfuncs:
                        costs_record.append(0)
                else:
                    logger.info("The two vectors are NOT the same!")
                    for cf in costfuncs:
                        logger.info("Compute the cost of this transformation of instance x [ID#{}] referring to [tree_id={};path_id={};path_length={}] using the \"{}\" function".
                                    format(i, tree_id, path_id, path_length, cf))
                        cf_val = compute_transformation_cost(x, x_prime, cf)
                        logger.info("The cost of this transformation of instance x [ID#{}] referring to [tree_id={};path_id={};path_length={}] using the \"{}\" function is {:.5f}".
                                    format(i, tree_id, path_id, path_length, cf, cf_val))
                        costs_record.append(cf_val)

                # append the record for the DataFrame of costs
                costs_rows.append(tuple(costs_record))
                # create the record made of instance id and the difference between
                # feature values
                signs_record.extend(np.subtract(x_prime, x))
                # append the record for the DataFrame of transformation signs
                signs_rows.append(tuple(signs_record))
        else:
            logger.info(
                "No positive transformation is associated with the true negative instance x [ID#{}]".format(i))

            costs_record = [i] + ['' for n in range(0, len(costs_header) - 1)]
            signs_record = [i] + ['' for n in range(0, len(signs_header) - 1)]

            #costs_record.extend(['' for n in range(0, len(costs_header) - 1)])
            costs_rows.append(tuple(costs_record))
            #signs_record.extend(['' for n in range(0, len(signs_header) - 1)])
            signs_rows.append(tuple(signs_record))

    logger.info("Finally, return the DataFrames containing the results")

    costs_df = pd.DataFrame(
        costs_rows, columns=costs_header).set_index('id')
    signs_df = pd.DataFrame(
        signs_rows, columns=signs_header).set_index('id')

    return costs_df, signs_df

##########################################################################


############################## Main ######################################

def main(options):

    logger.info("Loading original dataset from " + options['dataset_filename'])
    # Loading original dataset
    dataset = loading_dataset(options['dataset_filename'])

    logger.info("Loading transformed dataset from " +
                options['tweakings_filename'])
    # Loading transformed dataset
    tweaked_dataset = loading_dataset(options['tweakings_filename'])
    tweaked_dataset.set_index("id", inplace=True)

    logger.info(
        "Retrieving the portion of dataset indexed by the tweaked dataset")
    dataset = dataset[dataset.index.isin(tweaked_dataset.index)]

    logger.info(
        "Selecting only (true) negative instances from the portion of dataset")
    dataset = dataset[dataset["class"] == -1]

    logger.info(
        "Selecting features of only (true) negative instances from the portion of dataset indexed")
    # Features
    X = dataset.iloc[:, :len(dataset.columns) - 1]

    tweaked_costs_df, tweaked_signs_df = create_tweaked_costs_dataframe(
        X, tweaked_dataset, options['costfuncs'])

    costs_filename = options['tweakings_filename'].rsplit(".", 1)[
        0] + "_costs.tsv"
    costs_filename = costs_filename.split("/")[-1]

    signs_filename = options['tweakings_filename'].rsplit(".", 1)[
        0] + "_signs.tsv"
    signs_filename = signs_filename.split("/")[-1]

    logger.info("Save the computed costs of epsilon-transformations to {}".format(
        options['output_dirname'] + '/' + costs_filename))
    # Save costs of transformations
    tweaked_costs_df.to_csv(
        options['output_dirname'] + '/' + costs_filename, sep="\t", index=True)

    # Save signs of transformations
    logger.info("Save the computed signs of epsilon-transformations to {}".format(
        options['output_dirname'] + '/' + signs_filename))
    tweaked_signs_df.to_csv(
        options['output_dirname'] + '/' + signs_filename, sep="\t", index=True)


if __name__ == "__main__":
    sys.exit(main(get_options()))
