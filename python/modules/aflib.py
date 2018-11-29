#activation function library
import numpy as np

class activationFunction:
    def __init__(self):
        self.result = 0

    def sigmoid(self, w):
        return 1/(1+np.exp(-w))

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
    #"""Compute softmax values for each sets of scores in x."""
        #print("x : \n%s" % x)
        #print("np.max(x) : \n%s" % np.max(x))
        ex = np.exp(x-np.max(x))
        return ex / ex.sum(axis=0) # only difference
