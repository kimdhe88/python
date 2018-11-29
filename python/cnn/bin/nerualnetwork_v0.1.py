import numpy as np
import nnlib, fctlib
import time, os, tkinter

logicalGates = nnlib.neuralNetwork(input_nodes=2, output_nodes=2, learningrate=0.001, activefname="sigmoid")
logicalGates.add_hidden_layer(nodes = 5)

gateList = np.array(['AND', 'OR', 'NAND', 'XOR'])
inputs = np.array([ [0,0],[0,1],[1,0],[1,1] ])
targets = {
"AND":np.array([ [0],[0],[0],[1] ]),
"OR":np.array([ [0],[1],[1],[1] ]),
"NAND":np.array([ [1],[1],[1],[0] ]),
"XOR":np.array([ [0],[1],[1],[0] ])
}

logicalGates.start()
logicalGates.setData(inputs, targets["AND"])


def ANDgate():
   logicalGates.setData(inputs, targets["AND"], "AND")
def ORgate():
   logicalGates.setData(inputs, targets["OR"], "OR")
def NANDgate():
   logicalGates.setData(inputs, targets["NAND"], "NAND")
def XORgate():
   logicalGates.setData(inputs, targets["XOR"], "XOR")
def PS():
    logicalGates.start_painter(1,"show_progress")
def PE():
    logicalGates.stop_painter()

top = tkinter.Tk()
AND = tkinter.Button(top, text ="AND", command = ANDgate)
OR = tkinter.Button(top, text ="OR", command = ORgate)
NAND = tkinter.Button(top, text ="NAND", command = NANDgate)
XOR = tkinter.Button(top, text ="XOR", command = XORgate)
PS = tkinter.Button(top, text ="print start", command = PS)
PE = tkinter.Button(top, text ="print stop", command = PE)

AND.pack()
OR.pack()
NAND.pack()
XOR.pack()
PS.pack()
PE.pack()

top.mainloop()
