
from collections import OrderedDict
from lasagne.updates import get_or_compute_grads
from lasagne import utils

import numpy as np

import theano
import theano.tensor as T

def Adabound(loss_or_grads, params, lr=1e-3, beta1=0.9, beta2=0.999, final_lr=0.1, gamma=1e-3, 
        eps=1e-8, amsbound=False):  ## remove weight decay for convenience ... 

    all_grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()
    step_prev = theano.shared(utils.floatX(0.))

    one = T.constant(1)

    step = step_prev + 1
    bias_correction1 = one - beta1 ** step
    bias_correction2 = one - beta2 ** step
    
    lower_bound = final_lr * (one - one / (gamma * step + one))
    upper_bound = final_lr * (one + one / (gamma * step))

    for param, grad in zip(params, all_grads):

        value = param.get_value(borrow=True)

        exp_avg_prev = theano.shared(np.zeros_like(value), broadcastable=param.broadcastable)
        exp_avg_sq_prev = theano.shared(np.zeros_like(value), broadcastable=param.broadcastable)
        if amsbound:
            max_exp_avg_sq_prev = theano.shared(np.zeros_like(value), broadcastable=param.broadcastable)

        exp_avg = exp_avg_prev * beta1 + (one - beta1) * grad
        exp_avg_sq = exp_avg_sq_prev * beta2 + (one - beta2) * grad ** 2
        if amsbound:
            max_exp_avg_sq = T.maximum(max_exp_avg_sq_prev, exp_avg_sq)
            updates[max_exp_avg_sq_prev] = max_exp_avg_sq
            denom = T.sqrt(max_exp_avg_sq) + eps
        else:
            denom = T.sqrt(exp_avg_sq) + eps
        
        step_size = (lr * T.sqrt(bias_correction2) / bias_correction1) * np.ones_like(denom)
        step_size = T.clip((step_size / denom), lower_bound, upper_bound) * exp_avg

        updates[exp_avg_prev] = exp_avg
        updates[exp_avg_sq_prev] = exp_avg_sq
        updates[param] = param - step_size

    updates[step_prev] = step
    return updates



