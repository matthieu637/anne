# -*- coding: UTF-8 -*-
'''
Created on 20 fevr. 2012

@author: matthieu637
'''
from utils import randmm

default_file_rules = {'x':(False, 1),
                    '.': (False, 0)}

class DataFile():
    '''
    classdocs
    '''
    def __init__(self, name, bound=0, rules=default_file_rules):
        '''
        Constructor
        '''
        if(rules == default_file_rules):
            default_file_rules['.'] = (False, bound)
        
        self.inputs = []
        self.outputs = []
        self.rules = rules
        self._read_data(name)
        
    def _read_data(self, name):
        rfile = open(name, "r")
        
        line = rfile.readline()
        input_area = True
        nbr_example = 0
        while line != "":
            if(line[0] == '#'):
                pass
            elif(line[0:2] == '=>'):
                input_area = False
            elif(input_area):
                self._add_list(self.inputs, nbr_example, self._line_to_list(line))
            elif(line[0] == '\n'):
                input_area = True
                nbr_example += 1
            else:
                self._add_list(self.outputs, nbr_example, self._line_to_list(line))
            line = rfile.readline()

        rfile.close()
        
    def _line_to_list(self, line):
        lstring = list(line.replace(" ", "").replace("\n", ""))
                       
        result = []
        for sym in lstring:
            for key in self.rules:
                if(sym == key):
                    if(not self.rules[key][0]):
                        result += [self.rules[key][1]]
                    else:
                        result += [randmm(self.rules[key][1], self.rules[key][2])]
        return result
    
    def _add_list(self, llist, pos, data):
        if(len(llist) <= pos):
            llist.append([])
        llist[pos].extend(data)

if __name__ == '__main__':
    #
    #Some examples :
    #
    
    df = DataFile("data/digital_shape.txt", 0)
    print (df.inputs)
    print (df.outputs)


    #[[0, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0], ...
    #[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], ...
    
    r = {'x':(True, 0, 1),
         '.': (False, 0),
         '?': (True, 0, 0.02),
         '!': (False, 1)}
    df = DataFile("data/blindslight.txt", rules=r)
    print (df.inputs[0])
    print (df.outputs[0])
