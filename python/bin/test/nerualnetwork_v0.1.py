import numpy as np
import nnlib

logicalGates = nnlib.neuralNetwork(input_nodes=2, output_nodes=2, learningrate=0.1)
logicalGates.add_hidden_layer(nodes = 7)
#logicalGates.add_hidden_layer(nodes = 10)

gateList = np.array(['AND', 'OR', 'NAND', 'XOR'])
inputs = np.array([ [0,0],[0,1],[1,0],[1,1] ])
targets = {
"AND":np.array([ [0],[0],[0],[1] ]),
"OR":np.array([ [0],[1],[1],[1] ]),
"NAND":np.array([ [1],[1],[1],[0] ]),
"XOR":np.array([ [0],[1],[1],[0] ])
}

learncnt = 0

for gate in gateList:
    print('=====================================================================')
    print("1. [%s] gate result before learning" % gate)
    logicalGates.query(inputs)
    if learncnt == 0:
        logicalGates.train(inputs_list=inputs, targets_list=targets[gate])
    else:
        print("\n2. %s times progressed." % learncnt)
        logicalGates.train(inputs_list=inputs, targets_list=targets[gate], trglearncnt=learncnt)

    print("\n3. [%s] gate result after learning" % gate)
    logicalGates.query(inputs)
    print("=====================================================================\n\n")
