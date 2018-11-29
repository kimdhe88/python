#activation function library
import numpy as np

def getfunction(funcname="sigmoid"):
    if funcname == "sigmoid":
        return activationFunction.sigmoid
    elif funcname == "ReLU":
        return activationFunction.ReLU
    elif funcname == "LReLU":
        return activationFunction.LReLU
    elif funcname == "softmax":
        return activationFunction.softmax

class activationFunction:
    def __init__(self):
        self.rtn = 0
    #sample function
    def sample(x, deff=False):
        if deff:
            return 1
        else:
            return 1

    def sigmoid(x, deff=False):
        if deff:
            return activationFunction.sigmoid(x) * (1 - activationFunction.sigmoid(x))
        else:
            return 1/(1+np.exp(-x))

    def ReLU(x, deff=False):
        if deff:
            return 1. * (x > 0)
        else:
            return x * (x > 0)

    def LReLU(x, deff=False):
        if deff:
            return 1. * (x > 0) + 0.01 * (x < 0) + 0. * (x == 0)
            #return x * (x >= 0) + x * 0.01 * (x < 0)
        else:
            #return (x>=0)*x + (x<0)*0.01*x
            return x * (x >= 0) + x * 0.01 * (x < 0)

    def softmax(x):
    #"""Compute softmax values for each sets of scores in x."""
        #print("x : \n%s" % x)
        #print("np.max(x) : \n%s" % np.max(x))
        ex = np.exp(x-np.max(x))
        return ex / ex.sum(axis=0) # only difference
