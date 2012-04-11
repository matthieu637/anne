'''
Created on 11 avr. 2012

@author: matthieu
'''

from multiprocessing import Process, Value, RawValue
from math import exp
from ctypes import Structure, c_double

NB_CORE = 4

class T(Structure):
    _fields_ = [('q', c_double)]

def multi(i):
    i.q += 4

if __name__ == '__main__':
    t = T()
    t = RawValue(T, 3)
    p = Process(target=multi, args=(t,))
    p.start()
    p.join()
    print('finish')
    print(t.q)
