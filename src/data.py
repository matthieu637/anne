# -*- coding: UTF-8 -*-
'''
Created on 20 fevr. 2012

@author: matthieu637
'''

class DataFile():
    '''
    classdocs
    '''
    def __init__(self, name, bound=0):
        '''
        Constructor
        '''
        self.inputs = []
        self.outputs = []
        self.bound = bound
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
        lstring = list(line.replace(" ", "").replace("\n", "").replace(".", "0").replace("x", "1"))
        lint = [int(i) for i in lstring]
        return [ 1 if i == 1 else self.bound for i in lint]
    
    def _add_list(self, llist, pos, data):
        if(len(llist) <= pos):
            llist.append([])
        llist[pos].extend(data)

if __name__ == '__main__':
    #
    #Some examples :
    #
    
    
    df = DataFile("data/digital_shape.txt", -1)
    print (df.inputs)
    print (df.outputs)


    #[[0, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0], ...
    #[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], ...
