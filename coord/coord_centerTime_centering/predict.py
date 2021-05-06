#!/usr/bin/env python
import argparse 
import numpy as np
import theano
from theano import tensor as T
import lasagne
from tqdm import tqdm
from process import *


ap = argparse.ArgumentParser()
ap.add_argument("-t", "--test", required=True, type=str, help="file for test model")
ap.add_argument("-s", "--size", required=True, type=int, help="the size of model input")
ap.add_argument("-m", "--model", required=True, type=str, help="file name that store the model structure")
ap.add_argument("-p", "--params", required=True, type=str, help="file that stores the weights of model")
ap.add_argument("-o", "--output", required=True, type=str, help="define the file for the evaluation output")
args = vars(ap.parse_args())

eva = open(args['output'], 'w')
size = args['size']
model_file = args['model']
params = args['params']
test = args['test']


if __name__ == '__main__':

    import pkgutil
    loader = pkgutil.get_importer('../model')
    # load network from file in 'models' dir
    model = loader.find_module(model_file).load_module(model_file)

    input_var = T.tensor3('input')
    label_var= T.ivector('label')
    shape=(1, 3, size)
    
    net, _, _, _ = model.network(input_var, label_var, shape)

    # load saved parameters from "params"
    with open(params, 'rb') as f:
        import pickle
        params = pickle.load(f)
        lasagne.layers.set_all_param_values(net, params)
        pass

    output_var = lasagne.layers.get_output(net, deterministic=True)
    pred = theano.function([input_var], output_var)
    
    with open(test, "r") as TEST:
        for line in tqdm(TEST):
            table = line.strip().split("\t")
            eva.write("%s" % table[1])

            coord_pred = np.load(table[2])[:, [0, 2, 3]]
            # center normalization along with the timestamp
            coord_pred = centering(coord_pred)
            coord_pred = fix_size(coord_pred.T, size)

            coord_pred = coord_pred[:, :, np.newaxis].transpose((2, 0, 1)).astype("float32")

            predictions = pred(coord_pred)
            eva.write("\t%.4f\n" % predictions)

    
