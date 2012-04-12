'''
Created on 3 April 2012

@author: Matthieu Zimmer
'''

import matplotlib.pylab as plt

if __name__ == '__main__':
    
    x =  [ [1 for x in range(1935)] + 
          [2 for x in range(1060)] +
          [3 for x in range(932)] +
          [4 for x in range(876)] +
          [5 for x in range(865)] +
          [6 for x in range(879)] +
          [7 for x in range(971)] +
          [8 for x in range(1138)] +
          [9 for x in range(1338)] ]

    h, b, c = plt.hist(x, bins=9, histtype='bar')
    
    h = 100*h/float(len(x))
    
    plt.title("Distribution of the answer when the first-order is wrong")
    plt.ylabel("frequency")
    plt.xlabel("n th best active neuron")
    plt.show()
  
  
    
    
    x =  [ [1 for x in range(1570)] + 
          [2 for x in range(1083)] +
          [3 for x in range(902)] +
          [4 for x in range(902)] +
          [5 for x in range(1209)] +
          [6 for x in range(1877)] +
          [7 for x in range(1444)] +
          [8 for x in range(794)] +
          [9 for x in range(216)] ]

    h, b, c = plt.hist(x, bins=9, histtype='bar')
    
    h = 100*h/float(len(x))
    
    plt.title("Distribution of the answer when the first-order is wrong (1600 shapes)")
    plt.ylabel("frequency")
    plt.xlabel("n th best active neuron")
    plt.show()
    
    