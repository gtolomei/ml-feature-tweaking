#!/usr/bin/env python3
# encoding: utf-8

"""
extract_paths.py

Created by Gabriele Tolomei on 2019-04-10.
"""

import sys
import os
import argparse
import logging
import gzip
import numpy as np

from sklearn.externals import joblib
from sklearn import tree


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
        filename="./extract_paths.log", mode="w")
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
        description="""Extracts all the paths from an ensemble of decision trees (e.g., random forest), previously trained and serialized to disk.""")
    cmd_parser.add_argument(
        'model_filename',
        help="""Path to the file containing the serialized ensemble of decision trees (i.e., an instance of BaseEstimator class).""",
        type=str)
    cmd_parser.add_argument(
        'output_filename',
        help="""Path to the output file, which will contain all the paths extracted.""",
        type=str)
    args = cmd_parser.parse_args(cmd_args)

    options = {}
    options['model_filename'] = args.model_filename
    options['output_filename'] = args.output_filename

    return options


class decision_tree_structure:
    """
        Class storing the decision tree structure
    """

    def __init__(self, dot_string):
        self.dot_string = dot_string
        self.build_tree()

    def parse_link(self, line):
        """
            We assume line is a line describing
            a branch of the tree.

            Return the two nodes parent and child
        """
        line = line.rstrip(' ;')
        n1 = line.split(' -> ')[0]
        n2 = (line.split(' -> ')[1]).split(' ')[0]
        return (n1, n2)

    def parse_node(self, line):
        node_id = line.split(' ')[0]
        try:
            (left, right) = line.split(' [label="')
        except ValueError:
            print('Error splitting', line)
            sys.exit(-1)
        if right[0] == 'X':
            feature_id = ''
            i = 2
            while right[i] != ']':
                feature_id += right[i]
                i += 1
            i += 5
            feature_value = ''
            while right[i] != '\\':
                feature_value += right[i]
                i += 1
            return ('node', node_id, (feature_id, feature_value))
        else:
            (left, right) = line.split('nvalue = [')
            (vals, remainder) = right.split(']"')
            numbers = vals.split(',')
            numbers = list(map(lambda x: int(x.strip()), numbers))
            # update here for managing multiclass
            class_label = np.argmax(numbers)
            # if len(numbers) > 2:
            #     sys.exit(-1)
            # if numbers[0] > numbers[1]:
            #     class_label = '-1'
            # else:
            #     class_label = '1'
            return ('leaf', node_id, ('-1', class_label))

    def build_tree(self):
        """
            This method builds the tree that is
            later transversed to get all the paths
            that are positive (HI)

            DISCLAIMER:
            This version of build_tree has everything very
            hard-coded. It is impossible to run again if
            dot format changes.
        """
        self.root = 0
        self.nodes = {}
        self.links = {}
        for line in self.dot_string:
            line = line.rstrip()
            if '->' in line:
                # It's a link
                (from_node, to_node) = self.parse_link(line)
                try:
                    f = int(from_node)
                    t = int(to_node)
                except ValueError:
                    continue
                if f in self.links:
                    self.links[f].append(t)
                    if (self.links[f][0] > self.links[f][1]):
                        print('ouch')
                        sys.exit(-1)
                else:
                    self.links[f] = [t]
            else:
                # It's a node
                if ' [label="' not in line:
                    """ If the line does not contain [label=" it means is not a line
                        representing a node
                    """
                    continue
                (type, node_id, (f, v)) = self.parse_node(line)
                try:
                    node_id = int(node_id)
                except ValueError:
                    print('Error converting node_id... Please check', node_id)
                    continue
                if type == 'node':
                    try:
                        feature_id = int(f)
                        feature_value = float(v)
                    except ValueError:
                        continue
                    if node_id in self.nodes:
                        print('Duplicate node', node_id)
                        sys.exit(-1)
                    self.nodes[node_id] = (type, feature_id, feature_value)
                elif type == 'leaf':
                    try:
                        label_value = int(v)
                    except ValueError:
                        continue
                    if node_id in self.nodes:
                        print('Duplicate node', node_id)
                        sys.exit(-1)
                    self.nodes[node_id] = (type, label_value)
                else:
                    print('Unexpected error')
                    sys.exit(-1)

    def recursive_depth_first(self, root, current_path, k_paths, k):
        if self.nodes[root][0] == 'leaf':
            # We got to a leaf, add it to the list and return
            # the leaf in case it is k-labelled
            if self.nodes[root][1] == k:
                # It's a Hi path
                k_paths.append(current_path)
            return
        # It's not a leaf. Continue browsing :)

        # We decided to encode paths as list of triples as follows:
        # [0, [(14, <=, -0.7171), (7, >, 457.0), (12, <=, 54.609), (39, >, -0.059), ... ]
        # With this new encoding scheme we aim to represent the following:
        # - feature 14 needs to be less than or equal to -0.7171 (feature threshold is negative, direction is <=);
        # - feature 7 needs to be greater than 457.0 (feature threshold is positive, direction is >);
        # - feature 12 needs to be less than or equal to 54.609 (feature threshold is positive, direction is <=);
        # - feature 39 needs to greater than -0.059 (feature threshold is negative, direction is >);

        # First we go left
        current_node_content = (self.nodes[root][1],
                                "<=",
                                self.nodes[root][2]
                                )
        left_child = self.links[root][0]
        self.recursive_depth_first(left_child,
                                   current_path + [current_node_content],
                                   k_paths, k)

        # Then we go right
        current_node_content = (self.nodes[root][1],
                                ">",
                                self.nodes[root][2]
                                )
        right_child = self.links[root][1]
        self.recursive_depth_first(right_child,
                                   current_path + [current_node_content],
                                   k_paths, k)

    def get_k_leaved_paths(self, k):
        """
            This method returns all the paths that
            are labelled as `k`.
            The format is a list of pairs (feature,value)
            that stands for the check if feature <= value
        """
        list_of_k_leaved_paths = []
        current_path = []
        self.recursive_depth_first(
            self.root, current_path, list_of_k_leaved_paths, k)
        return list_of_k_leaved_paths


def load_model(model_filename):
    """
        This function loads a model from a dump done via scikit-learn
    """
    with open(model_filename, 'rb') as model_file:
        return joblib.load(model_file)


def enumerate_paths(model, tmp_filename='./model_paths.tmp'):

    logger = logging.getLogger(__name__)

    # dictionary used to store all the k-leavede paths extracted from the model
    # e.g., paths[k][tree_id] = [[(14, <=, -0.7171), (7, >, 457.0), (12, <=, 54.609), (39, >, -0.059)], ...]
    paths = {}
    classes = model.classes_  # extract all the leaf values

    logger.info("==> Start extracting all the paths of the ensemble ...")

    tree_id = 0  # start from tree_id 0
    for t in model.estimators_:  # loop through all the trees
        logger.info("Examining tree ID #{}".format(tree_id))
        tree.export_graphviz(t, out_file=tmp_filename)
        with open(tmp_filename, "r") as tmp:
            logger.debug(
                "Recreating tree structure of tree ID #{}".format(tree_id))
            dt_structure = decision_tree_structure(tmp.readlines())
            for k in classes:  # loop through all the class labels
                logger.debug(
                    "Extracting all the {}-leaved paths of tree ID #{}".format(k, tree_id))
                if k not in paths:  # check if the current class k has been already examined on a different tree
                    # if this is the first time we inspect class k, initialize the corresponding entry with an empty dictionary
                    paths[k] = {}
                paths[k][tree_id] = dt_structure.get_k_leaved_paths(k)
        tree_id += 1

    logger.info("==> Eventually, return all the paths extracted")

    return paths


def dump_paths(paths, output_filename):
    with gzip.GzipFile(output_filename + '.gz', 'wb') as output_file:
        joblib.dump(paths, output_file)


def main(options):
    logger = configure_logging(level=logging.INFO)

    # Load the serialized model (i.e., the tree ensemble)
    logger.info("==> Loading serialized model from `{}`".format(
        options['model_filename']))
    model = load_model(options['model_filename'])

    logger.info("*************** Model Info ***************")
    logger.info("n. of trees: {}".format(len(model.estimators_)))
    logger.info("n. of features: {}".format(model.n_features_))
    logger.info("classes: [{}]".format(
        ", ".join([str(c) for c in model.classes_])))
    logger.info("******************************************")

    # Extract all the (k-leaved) paths from the loaded tree ensemble
    logger.info("==> Extracting all paths from the just loaded model ...")
    paths = enumerate_paths(
        model, tmp_filename=options['model_filename'] + '.paths.tmp')

    for k in model.classes_:
        k_tot = 0
        for t in range(len(model.estimators_)):
            logger.debug(
                "Number of {}-leaved paths of tree ID #{}: {}".format(k, t, len(paths[k][t])))
            k_tot += len(paths[k][t])

        logger.info("Total number of {}-leaved paths: {}".format(k, k_tot))

    if os.path.isfile(options['model_filename'] + '.paths.tmp'):
        # Clear temporary path file
        logger.info("==> Cleaning up temporary path file `{}`".format(
            options['model_filename'] + '.paths.tmp'))
        os.remove(options['model_filename'] + '.paths.tmp')

    # Save all the k-leaved paths to disk
    logger.info("==> Saving all the extracted paths to `{}.gz`".format(
        options['output_filename']))
    dump_paths(paths, options['output_filename'])


if __name__ == '__main__':
    sys.exit(main(get_options()))
