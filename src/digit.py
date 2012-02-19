# -*- coding: UTF-8 -*-
'''
Created on 10 fevr. 2012

@author: matthieu637
'''

import numpy as np

class Factory:
    '''
    creates matrices who contains a graphical representation of digits
    '''
    @staticmethod
    def digitToMatrix(d, (n, m)=(5, 4), ini=0):
        '''
        returns a matrix(n,m) with graphical representation of d
        the background will be initialized at ini
        '''
        if d < 0 or d > 9 :
            raise Exception("%d is not a digit" % d)
        elif n < 2 or m < 2 :
            raise Exception("(%d,%d) is not enough" % n, m)
        else :
            mat = np.zeros((n, m)) + ini
            #right vertical line
            if d in (0, 1, 3, 8, 9):
                mat.T[m - 1] = np.ones(n)
                if d == 1:
                    tir = m - 1 - m / 3
                    mat[0][tir:m - 1] = np.ones(m - 1 - tir)
            #top, bottom and middle horizontal lines
            if d in (0, 2, 3, 5, 6, 8, 9):
                mat[0] = np.ones(m)
                if d != 0:
                    mat[n / 2] = np.ones(m)
                mat[n - 1] = np.ones(m)
                if d == 2:
                    mat.T[0][n / 2:n - 1] = np.ones(n - 1 - n / 2)
                    mat.T[m - 1][0:n / 2] = np.ones(n / 2)
                elif d in (0, 6, 8):
                    mat.T[0] = np.ones(n)
                elif d in (5, 9):
                    mat.T[0][0:n / 2] = np.ones(n / 2)
                if d in (5, 6):
                    mat.T[m - 1][n / 2:n - 1] = np.ones(n - 1 - n / 2)
            #middle horizontal line
            #TODO: implement a general method for 4 and 7
            if d in (4, 7):
                mat[n / 2] = np.ones(m)
                if d == 7:
                    mat[0] = np.ones(m)
                    mat[1][3] = 1
                    mat[3][1] = 1
                    mat[4][0] = 1
                else:
                    mat.T[0][0:m / 2] = np.ones(m / 2)
                    mat[1][2] = 1
                    mat[3][2] = 1
                    mat[4][2] = 1
            return mat
            
if __name__ == "__main__":
    '''
    displays the 10 digits
    '''
    print "\n\n".join(
                   [Factory.digitToMatrix(k, (5, 4)).__str__()
                    for k in range(10)]
                   )
