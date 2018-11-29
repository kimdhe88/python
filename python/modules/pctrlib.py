#perceptron library
import numpy as np
import time


class activationFunction:
    def __init__(self):
        self.result = 0

    def sigmoid(self, w):
        return 1/(1+np.exp(-w))

    #ReLU(Rectified Linear Unit, 렐루)
    def relu(self, x):
        return np.maximum(0, x)

#perceptron logicGates
class logicGates:
    def __init__(self):
        self.af = activationFunction()
        self.gateList = np.array(['AND', 'OR', 'NAND', 'XOR'])
        self.w = {
        "AND":np.random.uniform(-1.0,1.0,3),
        "OR":np.random.uniform(-1.0,1.0,3),
        "NAND":np.random.uniform(-1.0,1.0,3)
        }
        self.input = np.array([ [0,0],[0,1],[1,0],[1,1] ])
        self.output = {
        "AND":np.array([0,0,0,1]),
        "OR":np.array([0,1,1,1]),
        "NAND":np.array([1,1,1,0])
        }
        self.result = 0

        print('=====================================================================')
        print('\n[Start initial learning of AND, OR, and NAND gates.]\n\n')
        for gate in self.w.keys():
            print('# The initial weight value of the [%s] gate' % gate)
            print('[Before] w1 : %8.5f' % self.w[gate][0], ', w2 : %8.5f' % self.w[gate][1], ', bias : %8.5f\n' % self.w[gate][2])
            self.learn_gates(gate,0.01)
            print('[After ] w1 : %8.5f' % self.w[gate][0], ', w2 : %6.5f' % self.w[gate][1], ', bias : %8.5f\n\n' % self.w[gate][2])
        print('=====================================================================')


    def learn_gates(self, gate ,lnrt=0.01):
        failure_flag = 1
        while True:
            if failure_flag == 0:
                break
            failure_flag = 0
            for i in range(self.input.shape[0]):
                input = self.input[i]
                output = self.logicgates(gate, input)
                #Add bias data
                a_bias_to_input = np.insert(self.input[i], 2, 1)
                #print("a_bias_to_input : %s" % a_bias_to_input)
                PRDTD_output = self.output[gate][i]
                error = PRDTD_output - output
                if PRDTD_output != output:
                    failure_flag = 1
                self.w[gate] = self.w[gate] + lnrt * error * a_bias_to_input
        print('Learn of Gate [%s] is completed.' % gate)

    def logicgates(self, gate, input):
        if gate == 'XOR':
            s1 = self.logicgates('NAND', input)
            s2 = self.logicgates('OR', input)
            output = self.logicgates('AND',np.array([s1, s2]))
        else:
            a_bias_to_input = np.insert(input, 2, 1)
            output = int(round(self.af.sigmoid(self.w[gate].T.dot(a_bias_to_input))))
        return output
