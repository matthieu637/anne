# -*- coding: UTF-8 -*-
'''
Created on 20 February 2012

@author: Matthieu Zimmer
'''
from utils import randmm
from os import path as p

#see DataFile.__init__()
default_file_rules = {'x':(1,),
                    '.': (0,)}

class DataFile():
    '''
    Tool to read data from files 
    '''
    def __init__(self, path, bound=0, rules=default_file_rules):
        '''
        reads data from a file and allow you to use inputs / outputs lists
        path : path to the file 
        rules : dictionary to define the grammar of the file
        defines like :
          key            values
        symbol -> (fixed value,)
        symbol -> (min random value, max random value)
        '''
        if(rules == default_file_rules):
            rules['.'] = (bound,)
        
        self.inputs = []
        self.outputs = []
        self._rules = rules

        if(p.dirname(__file__) == ''):
            self._read_data(p.dirname(__file__) + "../data/" + path)
        else:
            self._read_data(p.dirname(__file__) + "/../data/" + path)
        
        if(len(self.inputs) != len(self.outputs)):
            print(self.inputs)
            raise Exception(path + " wrong data file")
        
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
            for key in self._rules:
                if(sym == key):
                    if(len(self._rules[key]) == 1):
                        result += [self._rules[key][0]]
                    else:
                        result += [randmm(self._rules[key][0], self._rules[key][1])]
        return result
    
    def _add_list(self, llist, pos, data):
        if(len(llist) <= pos):
            llist.append([])
        llist[pos].extend(data)

class DataFileR(DataFile):
    def _line_to_list(self, line):
        lfloat = []
        lstring = line.replace("\n", "").split(" ")
        for s in lstring:
            lfloat.append(float(s))
        return lfloat

if __name__ == '__main__':
    #
    #Some examples :
    #
    
    df = DataFile("digital_shape.txt", 0)
    print (df.inputs)
    print (df.outputs)


    #[[0, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0], ...
    #[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], ...
    
    r = {'x':(0, 1),
         '.': (0,),
         '?': (0, 0.02),
         '!': (1,)}
    df = DataFile("blindslight.txt", rules=r)
    print (df.inputs[0])
    print (df.outputs[0])
    
    
    df = DataFileR("iris.txt")
    print (df.inputs[10])
    print (df.outputs[10])
    
