# neuralNetwork Library
import numpy as np
import threading, time, os
import aflib, fctimerlib

class neuralNetwork(threading.Thread):
    def __init__(self, input_nodes, output_nodes, learningrate=0.01, activefname="sigmoid"):
        threading.Thread.__init__(self)
        self.af = aflib.getfunction(funcname=activefname)
        self.softmax = aflib.getfunction(funcname="softmax")

        self.nninfo = {}
        self.nninfo["layercnt"] = 2
        self.nninfo["ly0_nodes"] = input_nodes
        self.nninfo["ly1_nodes"] = output_nodes
        self.nninfo["outlayerpos"] = (self.nninfo["layercnt"] - 1)

        self.inodes = input_nodes
        self.onodes = output_nodes
        self.lr = learningrate
        self.w = {}

        self.init_weight()
        #self.w = {0:np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.inodes))}

        #parameters for adding hidden layers
        self.olayerpos = 1

        #parameters for learning counting
        self.learningcount = 0

        #for threading
        self.title = "learn Title"
        self.inputs_list = np.array([])
        self.targets_list = np.array([])

        #parameter for show status
        self.printflag = 0
        self.painter = 0

    def init_weight(self):
        for lylv in range(self.nninfo["layercnt"]):
            if lylv == 0:
                pass
            else:
                ply_nodes = self.nninfo["ly%s_nodes" % (lylv-1)]
                cly_nodes = self.nninfo["ly%s_nodes" % (lylv)]
                self.w["ly%s" % (lylv - 1)] = np.random.normal(0.0, pow(cly_nodes, -0.5), (cly_nodes, ply_nodes))
        #print(self.w)

    def add_hidden_layer(self, nodes):
        self.nninfo["ly%s_nodes" % (self.nninfo["layercnt"])] = self.nninfo["ly%s_nodes" % (self.nninfo["layercnt"] - 1)]
        self.nninfo["ly%s_nodes" % (self.nninfo["layercnt"] - 1)] = nodes
        self.nninfo["layercnt"] += 1
        self.nninfo["outlayerpos"] = (self.nninfo["layercnt"] - 1)
        self.init_weight()

    def run(self):
        while True:
            if self.checkData():
                self.learningcount += 1
                self.train()
            else:
                time.sleep(0.5)

    def checkData(self):
        return self.inputs_list.size

    def setData(self, inputs_list, targets_list, title=""):
        self.learningcount = 0
        self.title = title
        self.inputs_list = inputs_list
        self.targets_list = targets_list
        self.init_weight()

    def getData(self, datName):
        if datName == "lr":
            return self.lr
        elif datName == "lc":
            return self.learningcount
        elif datName == "w":
            return self.w

    def start_painter(self, interval, fcname):
        if fcname == "show_progress":
            self.painter = fctimerlib.functionTimer(interval,self.show_progress)
        self.painter.start()

    def stop_painter(self):
        self.painter.cancel()
        os.system('clear')

    def show_progress(self):
        os.system('clear')
        print("[%15s] learning count : %s" % (self.title, self.learningcount))
        self.query()

    #Learning by feedforward, backpropagation
    def train(self):
        if (self.inputs_list.size / self.inodes) != (self.targets_list.size / 1):
            print("\n[Err] The input list and the target list are different.\n")
            return 0
        for dataset in range(int((self.inputs_list.size / self.inodes))):
            self.backpropagation(inputs = np.array([self.inputs_list[dataset]]), targets = np.array([self.targets_list[dataset]]))

    #Overshooting setting function
    def query(self):
        for dataset in range(int((self.inputs_list.size / self.inodes))):
            fin_outputs = self.feedforward(inputs=np.array([self.inputs_list[dataset]]))
            rst = int(round(fin_outputs[0]))
            print("input : %s, output : [ %-0.6f ], result : [ %s ]" % (self.inputs_list[dataset], fin_outputs, rst))

    def feedforward(self, inputs, layerlevel=0):
        inputs = inputs.T
        if layerlevel == 0: #input layer
            out_c = self.af(inputs)
        else:
            net_c = np.dot(self.w["ly%s" % (layerlevel-1)], inputs)
            #print(net_c)
            out_c = self.af(net_c)

        if layerlevel != self.nninfo["outlayerpos"]:
            fin_outputs = self.feedforward(inputs = out_c.T, layerlevel = layerlevel + 1)
        else:
            fin_outputs = sum(out_c)
        return fin_outputs

    def backpropagation(self, inputs, targets, layerlevel=0):
        inputs = inputs.T

        if layerlevel == 0: #input layer
            net_c = inputs
        else:
            net_c = np.dot(self.w["ly%s" % (layerlevel-1)], inputs)

        out_c = self.af(net_c)
        if layerlevel != self.nninfo["outlayerpos"]:  #not out layer
            errors = self.backpropagation(inputs = out_c.T, targets = targets, layerlevel = layerlevel + 1)
        else:
            #Redistribute errors using soft max functions on the output layer
            error_total = targets - sum(out_c)
            #error distribution ratio
            errors_weight = self.softmax(out_c)
            errors = np.dot(errors_weight, error_total)

        #sigmoid differential
        delta_weight = errors * self.af(net_c, deff=True) * inputs.T
        #print(delta_weight)
        if layerlevel != 0:  # not input layer
            self.w["ly%s" % (layerlevel-1)] = np.add(self.w["ly%s" % (layerlevel-1)], delta_weight)
            next_errors = np.dot(self.w["ly%s" % (layerlevel-1)].T,errors)
        else:
            next_errors = 1

        return next_errors
