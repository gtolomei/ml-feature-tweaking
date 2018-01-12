#!/usr/bin/env python

from __future__ import print_function

import sys
import argparse

from sklearn.externals import joblib
from sklearn import tree


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
            print ('Error splitting', line)
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
            if len(numbers) > 2:
                sys.exit(-1)
            if numbers[0] > numbers[1]:
                class_label = '-1'
            else:
                class_label = '1'
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
                    #print ('Not really a node', line)
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

    def recursive_depth_first(self, root, current_path, hi_paths):
        if self.nodes[root][0] == 'leaf':
            # We got to a leaf, add it to the list and return
            # the leaf in case it is a Hi
            if self.nodes[root][1] == 1:
                # It's a Hi path
                hi_paths.append(current_path)
            return
        # It's not a leaf. Continue browsing :)

        # To fully characterize the path we consider the
        # following convention in path representation.
        # When the feature value is negative it means we
        # get to that node by going to the left child at
        # the parent node.
        # [0, [(14, -0.7171), (7, 457.0), (39, -0.059),... ]
        # For tree 0 when feature 14 is smaller than 0.7171 and
        # feature 7 is greater than 457 and feature 39 is smaller
        # than 0.059... etc.

        # Update to the convention above!!!
        # As we may have feature values that are already negatives
        # We decide to encode paths as list of triples as follows:
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
                                   hi_paths)

        # Then we go right
        current_node_content = (self.nodes[root][1],
                                ">",
                                self.nodes[root][2]
                                )
        right_child = self.links[root][1]
        self.recursive_depth_first(right_child,
                                   current_path + [current_node_content],
                                   hi_paths)

    def get_hi_paths(self):
        """
            This method returns all the paths that
            are labelled as hi.
            The format is a list of pairs (feature,value)
            that stands for the check if feature <= value
        """
        list_of_hi_paths = []
        current_path = []
        self.recursive_depth_first(self.root, current_path, list_of_hi_paths)
        return list_of_hi_paths


def parse_options():
    """
        This function parses the options passed from the command line.
        Run <thisscript> --help to learn how to invoke this script.
        Here are the options
        -m model_file
        -o output_file
    """
    parser = argparse.ArgumentParser(description='Extracts all the ' +
                                                 'positive (HI) paths ' +
                                                 'in a random forest model.')
    parser.add_argument('model', help='The name of the file ' +
                                      'containing the model')
    parser.add_argument('output', help='The name of the file ' +
                                       'containing the output')

    args = parser.parse_args()
    options = {}
    options['model'] = args.model
    options['output'] = args.output

    return options


def load_model(model_file_name):
    """
        This function loads a model from a dump done via scikit-learn
    """
    return joblib.load(model_file_name)


def enumerate_paths(model):
    paths = []
    tmp_file_name = '/tmp/tree.paths.dot'

    tree_id = 0
    for t in model.estimators_:
        tree.export_graphviz(t, out_file=tmp_file_name)
        with open(tmp_file_name, "r") as tmp:
            dt_structure = decision_tree_structure(tmp.readlines())
            for path in dt_structure.get_hi_paths():
                paths.append([tree_id, path])
        tree_id += 1

    return paths


def dump_paths(paths, output_file_name):
    with open(output_file_name, "w") as output_file:
        for path in paths:
            print(path, end='\n', file=output_file)


def main():
    options = parse_options()

    model = load_model(options['model'])

    hi_paths = enumerate_paths(model)

    dump_paths(hi_paths, options['output'])


if __name__ == '__main__':
    main()
