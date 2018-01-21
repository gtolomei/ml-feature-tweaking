#!/usr/bin/env python
# encoding: utf-8
"""
dump_recommendations.py

Created by Gabriele Tolomei on 2016-07-27.
Copyright (c) 2016 Yahoo! Labs. All rights reserved.
"""

import sys
import os
import argparse
import logging
import logging.handlers
import math
import ast
import json
import inspect
import collections
import cost_functions
import pandas as pd
import numpy as np
from scipy.stats import *
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
    "../log/dump_recommendations_" + str(os.getpid()) + ".log", mode='w', maxBytes=(1048576 * 5), backupCount=2, encoding=None, delay=0)
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
        'transformation_costs_filename',
        help="""Path to the input file containing the transformation costs.""",
        type=str)
    cmd_parser.add_argument(
        'transformation_signs_filename',
        help="""Path to the input file containing the transformation signs.""",
        type=str)
    cmd_parser.add_argument(
        'model_filename',
        help="""Path to the input file containing the learning model.""",
        type=str)
    cmd_parser.add_argument(
        'metadata_filename',
        help="""Path to the input file containing the ad metadata.""",
        type=str)
    cmd_parser.add_argument(
        'output_dirname',
        help="""Path to the output directory containing results.""",
        type=str)
    cmd_parser.add_argument(
        '-s',
        '--sort_by_cost',
        help="""List of cost keys used to sort transformations (e.g., cosine_distance).""",
        type=check_valid_cost_functions,
        default="cosine_distance")
    cmd_parser.add_argument(
        '-k',
        '--top_k',
        help="""Top-k transformations for each ad.""",
        type=check_valid_top_k,
        default=5)
    cmd_parser.add_argument(
        '-f',
        '--output_format',
        help="""Format used in the output recommendation file (e.g., json).""",
        type=check_valid_output_format,
        default="json")

    args = cmd_parser.parse_args(cmd_args)

    options = {}
    options['transformation_costs_filename'] = args.transformation_costs_filename
    options['transformation_signs_filename'] = args.transformation_signs_filename
    options['model_filename'] = args.model_filename
    options['metadata_filename'] = args.metadata_filename
    options['output_dirname'] = args.output_dirname
    options['sort_by_cost'] = args.sort_by_cost
    options['top_k'] = args.top_k
    options['output_format'] = args.output_format

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
            "No function in the input list [{}] is a valid cost function! Please, choose your input list from the following:\n{}".format(", ".join([cf for cf in input_cost_functions]), "\n".join(["* " + f for f in available_cost_functions])))

    if len(matched_cost_functions) < len(input_cost_functions):
        logger.info("The following input functions are not valid cost functions: [{}]".format(
            ", ".join([f for f in unmatched_cost_functions])))
        logger.info("The cost functions we will be using are the following: [{}]".format(
            ", ".join([f for f in matched_cost_functions])))

    return list(matched_cost_functions)


def check_valid_top_k(value):
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(
            "Input argument was {}. Please, choose an integer number which is equal to or greater than 1".format(ivalue))
    return ivalue


def check_valid_output_format(output_format):
    """
    This function is responsible for checking the validity of the specified output format of the recommendation file.

    Args:
        output_format (str): string representing the desired output format

    Return:
        The specified output format if this is allowed, an argparse.ArgumentTypeError otherwise
    """
    allowed_output_formats = set(["json", "xml", "csv", "tsv"])
    if output_format not in allowed_output_formats:
        raise argparse.ArgumentTypeError(
            "The specified output format was \"{}\" but this is NOT a valid format! Please, choose your output format from the following:\n{}".format(output_format, "\n".join(["* " + of for of in allowed_output_formats])))
    return output_format


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


def compute_feature_ranking(model):
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
                 axis=0)
    ranking = np.argsort(importances)[::-1]
    return ranking

##########################################################################

##########################################################################


def get_top_k_transformations_indices(transformation_costs, k, sort_by, group_by=["ad_id"]):
    grouped_transformations = transformation_costs.groupby(group_by)
    top_k_transformations_indices = {}
    for name, group in grouped_transformations:
        logger.info("Retrieve top-{} transformations for [{}] = [{}]".format(
            k, ",".join([gb for gb in group_by]), name))
        top_k_transformations_indices[int(name)] = group.sort_values(
            by=sort_by + ['path_length', 'unmatched_component_rate']).index[:k]
    return top_k_transformations_indices

##########################################################################

##########################################################################


def get_recommendations_from_transformation(transformation, feature_names, feature_ranking, no_recommendations_for):

    recommendations = []
    n = 1
    for i in feature_ranking:
        if feature_names[i] not in no_recommendations_for:
            recommendation = "increase"
            if transformation.get_value(feature_names[i]) < 0:
                recommendation = "decrease"
            if transformation.get_value(feature_names[i]) != 0:
                logger.info(" - {} {} [{}]".format(recommendation, feature_names[i].replace(
                    'Mobile_', '').replace('_FEATURE', ''), transformation.get_value(feature_names[i])))
                r_data = {}
                r_data["recommendation"] = {}
                r_data["recommendation"]["rank"] = n
                r_data["recommendation"]["feature"] = feature_names[i].replace(
                    'Mobile_', '').replace('_FEATURE', '')
                r_data["recommendation"]["sentence"] = 'Please, ' + recommendation + \
                    ' ' + \
                    feature_names[i].replace(
                        'Mobile_', '').replace('_FEATURE', '')
                r_data["recommendation"]["value"] = transformation.get_value(feature_names[
                                                                             i])
                recommendations.append(r_data)
                n += 1
    return recommendations

##########################################################################

##########################################################################


def generate_recommendations(transformation_signs, top_k_transformations_indices, k, feature_names, feature_ranking):

    recommendations = []

    for ad_id in top_k_transformations_indices:
        ad_transformations = {}
        ad_transformations["ad_id"] = ad_id
        ad_transformations["transformations"] = []

        transformation_indices = top_k_transformations_indices[ad_id]
        logger.info(
            "Generate recommendations for ad_id [{}] from the top-{} sets of transformations".format(ad_id, k))
        n = 1
        for i in transformation_indices:
            logger.info(
                "Generate recommendations using transformation #{}...".format(n))
            transformation = transformation_signs.ix[i][1:]
            t_recommendations = get_recommendations_from_transformation(transformation, feature_names, feature_ranking, no_recommendations_for=["Mobile_HISTORICAL_BOUNCE_RATE",
                                                                                                                                                "Mobile_HISTORICAL_DWELLTIME",
                                                                                                                                                "Mobile_HISTORICAL_DWELLTIME_CLICKS"
                                                                                                                                                ])
            # create an entry iff the list of recommendations generated from
            # this transformation is not empty!
            if t_recommendations:
                transformation_data = {}
                transformation_data["transformation"] = {}
                transformation_data["transformation"]["rank"] = n
                transformation_data["transformation"][
                    "recommendations"] = t_recommendations
                ad_transformations["transformations"].append(
                    transformation_data)

            n += 1

        recommendations.append(ad_transformations)

    return recommendations

##########################################################################


def save_recommendations_to_disk(recommendations, metadata, output_format, output_filename):

    if output_format == "json":
        save_recommendations_to_disk_as_json(
            recommendations, metadata, output_filename)
    if output_format == "xml":
        pass
        # save_recommendations_to_disk_as_xml(recommendations, metadata, output_filename)
    if output_format == "csv":
        save_recommendations_to_disk_as_text(
            recommendations, metadata, output_filename)
    # if ...
    #...
    if output_format == "tsv":
        save_recommendations_to_disk_as_text(
            recommendations, metadata, output_filename, sep="\t")

##########################################################################

##########################################################################


def save_recommendations_to_disk_as_json(recommendations, metadata, output_filename):
    with open(output_filename, 'w') as outfile:
        json.dump(recommendations, outfile)

##########################################################################

##########################################################################


def save_recommendations_to_disk_as_text(recommendations, metadata, output_filename, sep=","):
    out_fmt = "%s" + sep + "%s" + sep + "%s" + sep + \
        "%s" + sep + "%s" + sep + "%s" + sep + "%s" + "\n"
    header = "ad_id" + sep + "ad_title" + sep + "ad_description" + sep + "ad_sponsored_by" + sep + "ad_landing_page_url" + \
        sep + "transformation_rank" + sep + \
        sep.join(["recommendation_" + str(i) for i in range(1, 11)]) + "\n"
    with open(output_filename, 'w') as outfile:
        outfile.write(header)
        for entry in recommendations:
            ad_id = entry["ad_id"]
            ad_title = metadata.ix[entry["ad_id"]]["ad_title"]
            ad_description = metadata.ix[entry["ad_id"]]["ad_description"]
            ad_sponsored_by = metadata.ix[entry["ad_id"]]["ad_sponsored_by"]
            ad_img_url = metadata.ix[entry["ad_id"]]["ad_img_url"]
            ad_landing_page_url = metadata.ix[
                entry["ad_id"]]["ad_landing_page_url"]
            ad_transformations = entry["transformations"]
            for t in ad_transformations:
                transformation_rank = t["transformation"]["rank"]
                recs = t["transformation"]["recommendations"]
                rec_sentences = []
                for r in recs:
                    rec_sentences.append(r["recommendation"]["sentence"])
                if len(rec_sentences) < 10:
                    # padding with "NULL"
                    for i in range(len(rec_sentences)+1, 11):
                        rec_sentences.append("NULL")
                sentences = sep.join([s for s in rec_sentences])
                outfile.write(out_fmt % (ad_id, ad_title, ad_description, ad_sponsored_by,
                                         ad_landing_page_url, transformation_rank, sentences))


##########################################################################

############################## Main ######################################


def main(options):

    logger.info("Loading transformation costs dataset from " +
                options['transformation_costs_filename'])
    # Loading transformation costs
    transformation_costs = loading_dataset(
        options['transformation_costs_filename'])
    transformation_costs = transformation_costs[
        transformation_costs["tree_id"].notnull()]

    logger.info("Loading transformation signs dataset from " +
                options['transformation_signs_filename'])
    # Loading transformation signs
    transformation_signs = loading_dataset(
        options['transformation_signs_filename'])

    # Loading ad metadata
    metadata = loading_dataset(
        options['metadata_filename'])
    # Set the index to "ad_id"
    metadata = metadata.set_index("ad_id")
    # Select only the metadata for those ad ids interested by transformations
    metadata = metadata.ix[set(transformation_costs.ad_id)]

    logger.info("Loading model from " + options['model_filename'])
    # Loading model
    model = loading_model(options['model_filename'])

    logger.info("Compute feature ranking")
    # Compute feature ranking
    feature_ranking = compute_feature_ranking(model)

    logger.info(
        "Retrieve the indices to all top-{} transformations".format(options['top_k']))
    # Retrieve the indices to all the top-k transformations
    top_k_transformations_indices = get_top_k_transformations_indices(
        transformation_costs, k=options['top_k'], sort_by=options['sort_by_cost'], group_by=["ad_id"])

    logger.info("Retrieve the list of feature names")
    feature_names = transformation_signs.columns.values.tolist()[1:]

    logger.info("Generate all recommendations")
    # Generate the actual recommendations
    recommendations = generate_recommendations(transformation_signs, top_k_transformations_indices, options[
                                               'top_k'], feature_names, feature_ranking)

    recommendations_filename = 'recommendations_' + options['transformation_costs_filename'].split("_")[1] +\
        '_top_' + str(options['top_k']) + '_' +\
        '_'.join([cost for cost in options['sort_by_cost']]) +\
        '.' + options["output_format"]

    logger.info("Store generated recommendations into {} file located at {}".format(
        options["output_format"], options['output_dirname'] + '/' + recommendations_filename))
    # Save generated recommendations to disk
    save_recommendations_to_disk(recommendations, metadata, options["output_format"], options[
                                 'output_dirname'] + '/' + recommendations_filename)


if __name__ == "__main__":
    sys.exit(main(get_options()))
