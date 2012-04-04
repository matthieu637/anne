'''
Created on 3 April 2012

@author: Matthieu Zimmer
'''

import matplotlib.pylab as plt

if __name__ == '__main__':
    
    x =  [ [1 for x in range(2076)] + 
          [2 for x in range(1176)] +
          [3 for x in range(906)] +
          [4 for x in range(825)] +
          [5 for x in range(760)] +
          [6 for x in range(796)] +
          [7 for x in range(895)] +
          [8 for x in range(1114)] +
          [9 for x in range(1447)] ]

    h, b, c = plt.hist(x, bins=9, histtype='bar')
    
    h = 100*h/float(len(x))
    
    plt.title("Distribution of the answer when the first-order is wrong")
    plt.ylabel("frequency")
    plt.xlabel("n th best active neuron")
    plt.show()
    
    