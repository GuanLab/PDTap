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
from adabound import Adabound

# Parameters setting
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--train", type=str, required=True, help="file for training the model")
ap.add_argument("-s", "--size", type=int, required=True, help="length of the model input")
ap.add_argument("-m", "--model", type=str, required=True, help="model file")
ap.add_argument("-f", "--fold", type=int, required=True, help="fold number")
args = vars(ap.parse_args())

train_file = args['train']
size = args['size']  #2500
step = args['fold']  # 0 1 2 3 4
model_file = args['model']  # guan_version2_2500

channel = 2
batch_size = 8 
min_scale = 0.8
max_scale = 1.2
epoch_num = 51


# Generate data
def data_generator(paths, demos, labels, batch_size, is_train):
    
    assert paths.shape[0] == demos.shape[0] == labels.shape[0]

    for index in range(0, paths.shape[0], batch_size):

        if index + batch_size <= paths.shape[0]:
            batch_paths = paths[index:index+batch_size]
            batch_demos = demos[index:index+batch_size, :]
            batch_labels = labels[index:index+batch_size]
        else:
            return

        batch_vals = []
        for path in batch_paths:
            
            # some errors here
            try: 
                accel_data = np.load(path)[:, 2:]
            except:
                print(path)
                sys.exit()
            #######

            # No normalization
            # accel_data = normalization(accel_data)
            accel_data = fix_size(accel_data.T, size=size)

            if is_train:
                # time scaling
                scale = random.random() * (max_scale - min_scale) + min_scale
                accel_data = scale_time(accel_data, scale)

                # quarternion rotation
                theta = random.random() * math.pi * 2
                a = random.random()
                b = random.random()
                c = random.random()
                accel_data = quarternion_rotation(accel_data, theta, a, b, c)

                # scale_magnitude
                scales = [random.random() * (max_scale - min_scale) + min_scale for _ in range(3)]
                scale_magnitude(accel_data, scales)

            batch_vals.append(accel_data)
        
        batch_vals = np.asarray(batch_vals)

        # print("data shape: {}, demo shape: {}, label shape: {}".format(batch_vals.shape, batch_demos.shape, batch_labels.shape))

        yield batch_vals, batch_demos, batch_labels


def train(input_file, model_num, model_name):
    record = open(model_name, 'w')
    
    ## Split the training data into training and validation
    path_train, demo_train, path_validate, demo_validate, label_train, label_validate = train_validate_split(input_file, model_num, 0.5)
    
    ## Construct the network training framework
    input_var = T.tensor3('input')
    label_var = T.ivector('label')
    
    loader = pkgutil.get_importer('../model')
    model = loader.find_module(model_file).load_module(model_file)
    net, loss, te_loss, te_acc = model.network(input_var, label_var, [batch_size, channel, size])
    params = lasagne.layers.get_all_params(net, trainable=True)
    learning_rate = theano.shared(lasagne.utils.floatX(1e-4))
    updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)

    train_fn = theano.function([input_var, label_var], loss, updates=updates, allow_input_downcast=True)
    test_fn = theano.function([input_var, label_var], te_loss, allow_input_downcast=True)

    ## set these for storing the best parameters
    cur_epoch = 1
    best_epoch = 1
    best_test_error = 0x7fffffff

    print("Start training ...")

    while cur_epoch <= epoch_num:
        array_order = np.arange(path_train.shape[0])
        np.random.shuffle(array_order)

        train_error, test_error = 0, 0
        count = path_train.shape[0] // batch_size
        
        print("Batch data generating ... ")

        # without augmentation: set all is_train as False
        for (train_vals, train_demos, train_labels), (test_vals, test_demos, test_labels) \
                in tqdm(izip(data_generator(path_train[array_order], demo_train[array_order], label_train[array_order], batch_size, False), \
                data_generator(path_validate, demo_validate, label_validate, batch_size, False))):
            train_error += train_fn(train_vals, train_labels) / batch_size
            test_error += test_fn(test_vals, test_labels) / batch_size

        train_error /= count
        test_error /= count

        print("--------------- Fold:{} Model:{} Epoch:{} ---------------".format(step, model_num, cur_epoch))
        print(train_error, test_error)

        if test_error < best_test_error:
            best = [np.copy(p) for p in (lasagne.layers.get_all_param_values(net))]
            best_test_error = test_error
            best_epoch = cur_epoch

        if cur_epoch % 25 == 0:
            params_file = model_name + '_params_' + str(cur_epoch)
            with open(params_file, 'w') as PARAMS:
                pickle.dump(best, PARAMS)
            record.write('%d\t%d\t%.4f\t%.4f\n' % (cur_epoch, best_epoch, train_error, test_error))

        cur_epoch += 1

    record.close()


## train 5 models for one fold data
train(train_file, 1, "model1")
train(train_file, 2, "model2")
train(train_file, 3, "model3")
train(train_file, 4, "model4")
train(train_file, 5, "model5")
