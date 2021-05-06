#!/usr/bin/env python

import sys
import argparse

import math
import random
import pickle
import pkgutil
from itertools import izip
from tqdm import tqdm

import cv2
import numpy as np

import theano
import lasagne
from theano import tensor as T
from theano import shared

from process import *



def bound_parser(bound):

    # input format: "{{x1, y1}, {x2, y2}}"
    # only choose the first set because the second is always {104, 104}

    return eval(bound.replace("{", "(").replace("}", ")"))[0]


def bound_normalization(coords, bounds):

    # coords input format: [[buttonid, x, y], ...]
    # bounds input format: ["{{xl, yl}, {104, 104}}", "{{xr, yr}, {104, 104}}"]

    xl, yl = bound_parser(bounds[0])
    xr, yr = bound_parser(bounds[1])
    bound_map = {"TappedButtonLeft": (xl, yl), "TappedButtonRight": (xr, yr)}

    for record in coords:

        if record[0] == "TappedButtonNone":
            # actually, don't need to comapre the record, just compare the x-val with the bound
            if record[1] < xr:
                record[1] -= bound_map["TappedButtonLeft"][0]
                record[2] -= bound_map["TappedButtonLeft"][1]
            else:
                record[1] -= bound_map["TappedButtonRight"][0]
                record[2] -= bound_map["TappedButtonRight"][1]

        else:
            record[1] -= bound_map[record[0]][0]
            record[2] -= bound_map[record[0]][1]


input_file = "tapping_train.txt"
path_train, bound_train, path_validate, bound_validate, label_train, label_validate = train_validate_split(input_file, 1, 0.5)

sample = np.load(path_train[0])
bound = bound_train[0]
print("Head 10 rows: ", sample[:10, :])
print("Tail 10 rows: ", sample[-10:, :])
print("bounds: ", bound)

bound_normalization(sample[:, 1:], bound)
print("Head 10 rows: ", sample[:10, :])
print("Tail 10 rows: ", sample[-10:, :])

sample = fix_size(sample[:, 2:].T, 800)
print("Head 10 rows: ", sample[:10, :])
print("Tail 10 rows: ", sample[-10:, :])
print("bounds: ", bound)

