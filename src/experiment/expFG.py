'''
Created on 27 May 2012

@author: Matthieu Zimmer
'''

import numpy as np
import matplotlib.pyplot as plt
import os, sys, time

if __name__ == '__main__':
    
    data = np.loadtxt("expF1.data")
    data2 = np.loadtxt("expF2.data")
    data3 = np.loadtxt("expF3.data")
    data4 = np.loadtxt("expF4.data")
    data5 = np.loadtxt("expF5.data")
    data6 = np.loadtxt("expF6.data")
    
    data7 = np.loadtxt("expG1.data")
    data8 = np.loadtxt("expG2.data")
    points = list(range(len(data)))[::3]
    
    plt.plot(points, [data[i] for i in points], linewidth=3, label='F1')
    plt.plot(points, [data2[i] for i in points], linewidth=3, label='F2')
    plt.plot(points, [data3[i] for i in points], linewidth=3, label='F3')
    plt.plot(points, [data4[i] for i in points], linewidth=3, label='F4')
    plt.plot(points, [data5[i] for i in points], linewidth=3, label='F5')
    plt.plot(points, [data6[i] for i in points], linewidth=3, label='F6')
    plt.title('Classification performance of feedback networks')
    plt.ylabel('ERROR')
    plt.xlabel("EPOCHS")
    plt.legend(loc='best', frameon=False)
    plt.axis((0, len(data), 0, 1.01))         
    path = "/tmp/pyplot.%s.%s.png" % (sys.argv[0], time.strftime("%m-%d-%H-%M-%S", time.localtime()))
    plt.savefig(path)
    plt.show()
    
    
    
    
    plt.plot(points, [data[i] for i in points], linewidth=3, label='F1')
    plt.plot(points, [data7[i] for i in points], linewidth=3, label='G1')
    plt.plot(points, [data8[i] for i in points], linewidth=3, label='G2')
    plt.title('Classification performance of feedback networks')
    plt.ylabel('ERROR')
    plt.xlabel("EPOCHS")
    plt.legend(loc='best', frameon=False)
    plt.axis((0, len(data), 0, 1.01))         
    path = "/tmp/pyplot.%s.%s.png" % (sys.argv[0], time.strftime("%m-%d-%H-%M-%S", time.localtime()))
    plt.savefig(path)
    plt.show()
    
