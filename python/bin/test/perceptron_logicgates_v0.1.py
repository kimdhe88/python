import pctrlib
import numpy as np

lg = pctrlib.logicGates()


#input = np.array([0,1])
#lg.start_learning(lnrt=0.01)
input = np.array([ [0,0], [0,1], [1,0], [1,1] ])
gateList = np.array(['AND','OR','NAND','XOR'])

print('\n\n\n[ Confirm learning results ]\n')
for gate in gateList:
    print('# Checking [%s] Gate Results\n' % gate)
    for cinput in input:
        print('input : %-10s' % cinput, 'output : ', lg.logicgates(gate, cinput))
    print('\n')
