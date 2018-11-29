# neuralNetwork Library
import numpy as np
import types
import aflib

class neuralNetwork:
    def __init__(self, input_nodes, output_nodes, learningrate=0.01):
        self.af = aflib.activationFunction()
        self.inodes = input_nodes
        self.onodes = output_nodes
        self.lr = learningrate
        self.w = {0:np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.inodes))}

        #parameters for overshooting
        self.ovsflag = 0
        self.ovsorirate = self.lr
        self.ovsrcnt = 0
        self.ovsmaxcnt = 0

        #parameters for adding hidden layers
        self.layercnt = 2
        self.ilayerpos = 0
        self.olayerpos = 1

        #parameters for termination of learning
        self.threshold = 0.2
        self.errorvalue = 0.0

        #parameters for learning counting
        self.learningcount = 0

        #parameter for dump
        self.dumpflag = 0

    def add_hidden_layer(self, nodes):
        targetlayerpos = self.olayerpos
        prevlayerpos = self.olayerpos - 1
        self.layercnt += 1
        self.olayerpos += 1

        self.w[prevlayerpos] = np.random.normal(0.0, pow(nodes, -0.5), (nodes, self.w[prevlayerpos].shape[1]))
        self.w[targetlayerpos] = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, nodes))

    def show_nodes(self):
        print(self.w)

    #Learning by feedforward, backpropagation
    def train(self, inputs_list, targets_list, trglearncnt=0):
        inputs_num = inputs_list.shape[0]
        targets_num = targets_list.shape[0]

        if inputs_num != targets_num:
            print("\n[Err] The input list and the target list are different.\n")
            return 0

        if trglearncnt == 0:
            self.learningcount = 0
            while True:
                break_flag = 1
                self.learningcount += 1
                if self.learningcount % 10000 == 0:
                    self.setupovs(ovsmaxcnt=1000, ovsrate=0.9)
                    print("current learningcount [%s]" % self.learningcount)



                self.checkovs()
                for dataset in range(inputs_num):
                    self.backpropagation(inputs = np.array([inputs_list[dataset]]), targets = np.array([targets_list[dataset]]))
                    if self.learningcount >= 100000 and self.learningcount % 10000 == 0 or self.learningcount % 10000 == 0 and self.dumpflag:
                        print("target : %s, self.errorvalue : [%0.10f]" % (targets_list[dataset], self.errorvalue))
                    if self.errorvalue >= self.threshold:
                        break_flag = 0

                if break_flag == 1:
                    print("The total has been learned [ %s ] times." % self.learningcount)
                    self.unsetovs()
                    break
        else:
            for learningcount in range(trglearncnt):
                if learningcount % 10000 == 0:
                    print("currnet learning count : %s" % cnum)
                for dataset in range(inputs_num):
                    self.backpropagation(inputs = np.array([inputs_list[dataset]]), targets = np.array([targets_list[dataset]]))

    #Overshooting setting function
    def setupovs(self, ovsmaxcnt=500, ovsrate=0.5):
        self.ovsflag = 1
        self.lr = ovsrate
        #print("learningrate change : %s" % self.lr)
        self.ovsrcnt = 0
        self.ovsmaxcnt = ovsmaxcnt

    def checkovs(self):
        if self.ovsflag == 0:
            return 0

        self.ovsrcnt += 1
        if self.ovsrcnt >= self.ovsmaxcnt:
            self.ovsflag = 0
            self.lr = self.ovsorirate
            #print("learningrate recive : %s" % self.lr)

    def unsetovs(self):
        self.ovsflag = 0
        self.lr = self.ovsorirate
        #print("learningrate recive : %s" % self.lr)

    def query(self, inputs_list):
        inputs_num = inputs_list.shape[0]
        for dataset in range(inputs_num):
            fin_outputs = self.feedforward(inputs=np.array([inputs_list[dataset]]))
            rst = int(round(fin_outputs[0]))
            print("input : %s, output : [ %-0.6f ], result : [ %s ]" % (inputs_list[dataset], fin_outputs, rst))

    def feedforward(self, inputs, depth=0):
        inputs = inputs.T
        if depth == self.ilayerpos:
            out_c = self.af.sigmoid(inputs)
        else:
            net_c = np.dot(self.w[depth-1], inputs)
            out_c = self.af.sigmoid(net_c)

        if depth != self.olayerpos:
            fin_outputs = self.feedforward(inputs = out_c.T, depth = depth + 1)
        else:
            fin_outputs = sum(out_c)

        return fin_outputs

    def backpropagation(self, inputs, targets, depth=0):
        inputs = inputs.T

        if depth == self.ilayerpos:
            net_c = inputs
        else:
            net_c = np.dot(self.w[depth-1], inputs)
        out_c = self.af.sigmoid(net_c)

        if depth != self.olayerpos:
            errors = self.backpropagation(inputs = out_c.T, targets = targets, depth = depth + 1)
        else:
            #Redistribute errors using soft max functions on the output layer
            error_total = targets - sum(out_c)
            #error distribution ratio
            errors_weight = self.af.softmax(out_c)
            errors = np.dot(errors_weight, error_total)
            self.errorvalue = abs(error_total)
            #if self.learningcount % 10000 == 0 and self.dumpflag:
            #    print("target : %s, error_total : [%0.10f], errors : \n%s" % (targets, error_total, errors))

        delta_weight = errors * out_c * (1 - out_c) * inputs.T
        if depth != self.ilayerpos:
            self.w[depth-1] = np.add(self.w[depth-1], delta_weight)
            next_errors = np.dot(self.w[depth-1].T,errors)
        else:
            next_errors = 1

        return next_errors
