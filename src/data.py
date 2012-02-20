# -*- coding: UTF-8 -*-
'''
Created on 20 fevr. 2012

@author: matthieu637
'''

class DataFile():
    '''
    classdocs
    '''


    def __init__(self, name):
        '''
        Constructor
        '''
        self.inputs = []
        self.outputs = []
        self._read_data(name)
        
    def _read_data(self, name):
        file = open(name, "r")
        
        line = file.readline()
        while line != "":
            if(line[0]=='#'):
                line=file.readline()
                continue
            elif(line[0:1]=="=>"):
                line=file.readline()
                continue
            print line.replace(" ", "").replace(".","0").replace("x","1").split()

            line=file.readline()
        
        
        print self.inputs
        print self.outputs
        
        file.close()

if __name__ == '__main__':
    #
    #Some examples :
    #
    
    
    df = DataFile("data/digital_shape.txt")
    