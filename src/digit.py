# -*- coding: UTF-8 -*-
'''
Created on 10 fevr. 2012

@author: matthieu637
'''

import numpy as np

class Factory:
    '''
    create matrix who contains a graphical representation of digits
    '''
    @staticmethod
    def digitToMatrix(d, (n,n) = (5,4), dtype = np.int_):
        '''
        return a matrix(n,n) with graphical representation of d
        ''' 
        if d < 0 or d > 9 :
            raise Exception("%d is not a digit" % d)
        else :
            #TODO : shape for 0
            mat = np.zeros((n,n), dtype)
            #right vertical line
            if d in (1,3,8,9):
                mat.T[n-1] = np.ones(n, dtype)
                if d == 1:
                    tir = n-1-n/3
                    mat[0][tir:n-1]= np.ones(n-1-tir, dtype)
            #top, bottom and middle horizontal lines
            if d in (2,3,5,6,8,9):
                mat[0] = np.ones(n, dtype)
                mat[n/2] = np.ones(n, dtype)
                mat[n-1] = np.ones(n, dtype)
                if d == 2:
                    mat.T[0][n/2:n-1]= np.ones(n-1-n/2, dtype)
                    mat.T[n-1][0:n/2]= np.ones(n/2, dtype)
                elif d in (6,8):
                    mat.T[0] = np.ones(n, dtype)
                elif d in (5, 9):
                    mat.T[0][0:n/2]= np.ones(n/2, dtype)
                if d in (5,6):
                    mat.T[n-1][n/2:n-1]= np.ones(n-1-n/2, dtype)
            #middle horizontal line
            if d in (4, 7):
                mat[n/2] = np.ones(n, dtype)
                if d== 7:
                    mat[0] = np.ones(n, dtype)
                    mat[1][3] = 1
                    mat[3][1] = 1
                    mat[4][0] = 1
                else:
                    mat.T[0][0:n/2]= np.ones(n/2, dtype)
                    mat[1][2] = 1
                    mat[3][2] = 1
                    mat[4][2] = 1
            return mat
            
if __name__ == "__main__":
    print "\n\n".join(
                   [Factory.digitToMatrix(k, (5, 4), np.int_).__str__() 
                    for k in range(10)]
                   )

    