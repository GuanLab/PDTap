from theano import tensor as T
import lasagne as nn
from lasagne.layers import batch_norm as bn
from lasagne.layers import DenseLayer

#def sorenson_dice(pred, tgt, ss=10):

def network(input_var, label_var, shape):
    layer = nn.layers.InputLayer(shape, input_var)                          # 800
    layer = bn(nn.layers.Conv1DLayer(layer, num_filters=8, filter_size=5))  # 796
    layer = nn.layers.MaxPool1DLayer(layer, pool_size=2)                    # 398
    layer = bn(nn.layers.Conv1DLayer(layer, num_filters=16, filter_size=5)) # 394
    layer = nn.layers.MaxPool1DLayer(layer, pool_size=2)                    # 197
    layer = bn(nn.layers.Conv1DLayer(layer, num_filters=32, filter_size=4)) # 194
    layer = nn.layers.MaxPool1DLayer(layer, pool_size=2)                    # 97
    layer = bn(nn.layers.Conv1DLayer(layer, num_filters=32, filter_size=4)) # 94
    layer = nn.layers.MaxPool1DLayer(layer, pool_size=2)                    # 47
    layer = bn(nn.layers.Conv1DLayer(layer, num_filters=64, filter_size=4)) # 44
    layer = nn.layers.MaxPool1DLayer(layer, pool_size=2)                    # 22
    layer = bn(nn.layers.Conv1DLayer(layer, num_filters=64, filter_size=5)) # 18
    layer = nn.layers.MaxPool1DLayer(layer, pool_size=2)                    # 9
    layer = DenseLayer(layer, num_units=1,nonlinearity=nn.nonlinearities.sigmoid) 
    output = nn.layers.get_output(layer).flatten().clip(0.00001,0.99999)
    output_det = nn.layers.get_output(layer, deterministic=True).flatten().clip(0.00001,0.99999)
    loss = nn.objectives.binary_crossentropy(output, label_var).mean() #, ss=ss)
    te_loss = nn.objectives.binary_crossentropy(output_det, label_var).mean() #,ss=ss)
    te_acc = nn.objectives.binary_accuracy(output_det, label_var).mean()

    return layer, loss, te_loss, te_acc

