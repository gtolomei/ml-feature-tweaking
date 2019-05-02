#!/usr/bin/env python
# encoding: utf-8
"""
distance_functions.py

Created by Gabriele Tolomei on 2019-05-02.
"""

import sys
import logging
import logging.handlers
import numpy as np
from scipy import spatial
from scipy.stats import *

# console logging format
CONSOLE_LOGGING_FORMAT = '%(asctime)-15s *** %(levelname)s *** %(message)s'
# file logging format
FILE_LOGGING_FORMAT = '%(asctime)-15s *** %(levelname)s [%(filename)s:%(lineno)s - %(funcName)s()] *** %(message)s'

# get the root logger
logger = logging.getLogger(__name__)
# set the logging level (default: DEBUG)
logger.setLevel(logging.INFO)
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
    "./logs/distance_functions.log",
    mode='w',
    maxBytes=(
        1048576 * 5),
    backupCount=2,
    encoding=None,
    delay=0)
# set the file handler logging format
file_logging_format = logging.Formatter(FILE_LOGGING_FORMAT)
# specify the logging format for this file handler
file_handler.setFormatter(file_logging_format)
# set the logging level for this file handler (default: DEBUG)
file_handler.setLevel(logging.INFO)
# attach this file handler to the logger
logger.addHandler(file_handler)

##########################################################################

######################################## Normalize vector ################


def __normalize_vector(v):
    """
    This function checks wheter the given input vector is already normalized (i.e. if its L-2 norm is 1).
    If it does then the function just returns the input vector as it is, otherwise it normalizes it before
    returning it.
    Firstly, however, it also transforms the vector into a numpy.array if needed.

    Args:
        v (any sequence): the input vector

    Returns:
        v' (numpy.array): the normalized vector v' as a numpy.array
    """

    logger.debug("Transform the input vector into a numpy array if needed")
    v = np.asarray(v)
    logger.debug(
        "Compute the L-2 norm (i.e. Frobenius norm) of the input vector")
    norm = np.linalg.norm(v)
    if norm == 1:
        logger.debug(
            "The L-2 norm (i.e. Frobenius norm) of the input vector is already {:.1f}. Let's just return the input vector as it is!".format(norm))
        return v
    logger.debug(
        "The L-2 norm (i.e. Frobenius norm) of the input vector is {:.3f}. Let's normalize the input vector before returning it!".format(norm))
    return v / norm

##########################################################################

# Count matches between two vect


def __count_matches(u, v):
    """
    This function returns the number of matching elements between two input vectors u and v.
    (Assumption: len(u) = len(v))

    Args:
        u (any sequence): first vector
        v (any sequence): second vector

    Returns:
        int: the number of matching elements between u and v
    """
    return len([i for i, j in zip(u, v) if i == j])

##########################################################################

# Count unmatches between two ve


def __count_unmatches(u, v):
    """
    This function returns the number of unmatching elements between two input vectors u and v.
    (Assumption: len(u) = len(v))

    Args:
        u (any sequence): first vector
        v (any sequence): second vector

    Returns:
        int: the number of unmatching elements between u and v
    """
    return len([i for i, j in zip(u, v) if i != j])

##########################################################################

# Compute the Unmatched Component Rate (UCR) between two


def unmatched_component_rate(u, v):
    """
    This function measures the distance between two input vectors u and v in terms of the ratio of
    different (i.e. unmatching) components, assuming len(u) = len(v)

    Example:
    u = [1, 2, 3, 4, 5, 4, 2, 2, 7, 10]
    v = [9, 2, 7, 6, 5, 4, 7, 6, 11, 7]
    len(u) = len(v) = 10
    unmatched_components = {i | u[i] != v[i]} = {0, 2, 3, 6, 7, 8, 9}
    unmatched_component_rate = |unmatched_components|/len(u) = 7/10 = 0.7

    Args:
        u (any sequence): first vector
        v (any sequence): second vector

    Returns:
        float: the ratio of unmatched components between u and v, normalized by the length of the vectors
    """

    logger.debug(
        "Compute the unmatched component rate (UCR) between the two input vectors u and v")
    return __count_unmatches(u, v) / float(len(u))

##########################################################################

# Compute the Euclidean Distance between tw


def euclidean_distance(u, v, normalize=False):
    """
    This function returns the euclidean distance between two vectors u and v.

    Args:
        u (any sequence): first vector
        v (any sequence): second vector

    Returns:
        float: the euclidean distance between u and v computed as the L-2 norm of the vector
                resulting from the difference (u - v)
    """
    if normalize:
        logger.debug("Normalize the two input vectors")
        u = __normalize_vector(u)
        v = __normalize_vector(v)
    logger.debug(
        "Compute the Euclidean distance between the two input vectors u and v")
    return np.linalg.norm(u - v)

##########################################################################

# Compute the Cosine Distance between two v


def cosine_distance(u, v, normalize=False):
    """
    This function returns the cosine distance between two vectors u and v.
    Invariant with respect to the magnitude of the vectors (i.e. scaling)

    Args:
        u (any sequence): first vector
        v (any sequence): second vector

    Returns:
        float: the cosine distance between u and v
    """
    if normalize:
        logger.debug("Normalize the two input vectors")
        u = __normalize_vector(u)
        v = __normalize_vector(v)

    logger.debug(
        "Compute the Cosine distance between the two input vectors u and v")
    return spatial.distance.cosine(u, v)

##########################################################################

# Compute the Jaccard Distance between two


def jaccard_distance(u, v):
    """
    This function returns the Jaccard distance between two vectors u and v.
    The Jaccard index Js defines a similarity measure; our distance measure is obtained as Jd = 1 - Js

    Example:
    u = [1, 2, 3, 4, 5, 4, 2, 2, 7, 10]
    v = [9, 2, 7, 6, 5, 4, 7, 6, 11, 7]
    len(u) = len(v) = 10
    set(u) = {1, 2, 3, 4, 5, 7, 10} is the set of unique values of u (therefore disregarding order and multiplicity)
    set(v) = {9, 2, 7, 6, 5, 4, 11} is the set of unique values of v (therefore disregarding order and multiplicity)
    len(set(u)) = 7
    len(set(v)) = 7
    set(u and v) = {2, 4, 5, 7}
    set(u or v) = {1, 2, 3, 4, 5, 6, 7, 9, 10, 11}
    len(set(u and v)) = 4
    len(set(u or v)) = 10

    Js = len(set(u and v))/len(set(u or v)) = 4/10 = 0.4
    Jd = 1 - Js = 1 - 0.4 = 0.6

    Args:
        u (any sequence): first vector
        v (any sequence): second vector

    Returns:
        float: the Jaccard distance between u and v
    """

    logger.debug(
        "Compute the Jaccard distance between the two input vectors u and v")
    logger.debug("Transform both vectors into sets of elements")
    s_u = set(u)
    logger.debug("First input vector u has {} unique values".format(len(s_u)))
    s_v = set(v)
    logger.debug("Second input vector v has {} unique values".format(len(s_v)))
    s_u_and_v = s_u.intersection(s_v)
    logger.debug(
        "The intersection set of the two vectors has {} unique values".format(
            len(s_u_and_v)))
    s_u_or_v = s_u.union(s_v)
    logger.debug(
        "The union set of the two vectors has {} unique values".format(
            len(s_u_or_v)))
    Js = len(s_u_and_v) / float(len(s_u_or_v))
    logger.debug(
        "Jaccard similarity index between u and v is Js = {:.5f}".format(Js))
    Jd = 1 - Js
    logger.debug(
        "Finally, return the Jaccard distance index between u and v = (1 - Js) = (1 - {:.5f}) = {:.5f}".format(Js, Jd))

    return Jd

##########################################################################

# Compute the Pearson Correlation distance


def pearson_correlation_distance(u, v):
    """
    This function computes the distance between two input vectors u and v
    in terms of their Pearson's correlation coefficient.
    This coefficient is invariant to the magnitude of the vectors (i.e. scaling)
    and also to adding any constant to all elements

    Args:
        u (any sequence): first vector
        v (any sequence): second vector

    Returns:
        float: the Pearson's correlation distance between u and v

    """

    logger.debug(
        "Compute the Pearson's correlation coefficient rho between the two input vectors u and v")
    rho = stats.pearsonr(u, v)[0]
    logger.debug(
        "Pearson's correlation coefficient between u and v is rho = {:.5f}".format(rho))
    rho_d = 1 - rho
    logger.debug(
        "Finally, return the Pearson's correlation distance between u and v = (1 - rho) = (1 - {:.5f}) = {:.5f}".format(rho, rho_d))

    return rho_d
