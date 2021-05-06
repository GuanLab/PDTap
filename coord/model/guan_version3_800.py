from theano import tensor as T
import lasagne as nn
from lasagne.layers import batch_norm as bn
from lasagne.layers import DenseLayer

#def sorenson_dice(pred, tgt, ss=10):

def network(input_var, label_var, shape):
    layer = nn.layers.InputLayer(shape, input_var)                          # 800
    
    layer = bn(nn.layers.Conv1DLayer(layer, num_filters=8, filter_size=5, pad="same"))  
    layer = bn(nn.layers.Conv1DLayer(layer, num_filters=8, filter_size=5, pad="same"))  
    layer = nn.layers.MaxPool1DLayer(layer, pool_size=2)                    # 400
    
    layer = bn(nn.layers.Conv1DLayer(layer, num_filters=16, filter_size=5, pad="same")) # 394
    layer = bn(nn.layers.Conv1DLayer(layer, num_filters=16, filter_size=5, pad="same")) # 394
    layer = nn.layers.MaxPool1DLayer(layer, pool_size=2)                    # 200
    
    layer = bn(nn.layers.Conv1DLayer(layer, num_filters=32, filter_size=3, pad="same")) # 194
    layer = bn(nn.layers.Conv1DLayer(layer, num_filters=32, filter_size=3, pad="same")) # 194
    layer = nn.layers.MaxPool1DLayer(layer, pool_size=2)                    # 100
    
    layer = bn(nn.layers.Conv1DLayer(layer, num_filters=32, filter_size=3, pad="same")) # 94
    layer = bn(nn.layers.Conv1DLayer(layer, num_filters=32, filter_size=3, pad="same")) # 94
    layer = nn.layers.MaxPool1DLayer(layer, pool_size=2)                    # 50
    
    layer = bn(nn.layers.Conv1DLayer(layer, num_filters=64, filter_size=3, pad="same")) # 44
    layer = bn(nn.layers.Conv1DLayer(layer, num_filters=64, filter_size=3, pad="same")) # 44
    layer = nn.layers.MaxPool1DLayer(layer, pool_size=2)                    # 25
    
    layer = bn(nn.layers.Conv1DLayer(layer, num_filters=128, filter_size=4)) # 22
    layer = nn.layers.MaxPool1DLayer(layer, pool_size=2)                    # 11
    
    layer = DenseLayer(layer, num_units=1,nonlinearity=nn.nonlinearities.sigmoid) 
    
    output = nn.layers.get_output(layer).flatten().clip(0.00001,0.99999)
    output_det = nn.layers.get_output(layer, deterministic=True).flatten().clip(0.00001,0.99999)
    
    loss = nn.objectives.binary_crossentropy(output, label_var).mean() #, ss=ss)
    te_loss = nn.objectives.binary_crossentropy(output_det, label_var).mean() #,ss=ss)
    te_acc = nn.objectives.binary_accuracy(output_det, label_var).mean()

    return layer, loss, te_loss, te_acc

