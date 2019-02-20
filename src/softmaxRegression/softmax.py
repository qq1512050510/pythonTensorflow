#coding:utf-8
'''
Created on 2019年2月19日

@author: adp
@pdf:Machine Learning with TensorFlow MEAP v10.pdf
@page:95
'''
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

#B
x1_label0 = np.random.normal(1,1,(100,1))
x2_label0 = np.random.normal(1,1,(100,1))
#C
x1_label1 = np.random.normal(5,1,(100,1))
x2_label1 = np.random.normal(4,1,(100,1))
#D
x1_label2 = np.random.normal(8,1,(100,1))
x2_label2 = np.random.normal(0,1,(100,1))

plt.scatter(x1_label0,x2_label0,c='r',marker='o',s=60)
plt.scatter(x1_label1,x2_label1,c='g',marker='x',s=60)
plt.scatter(x1_label2,x2_label2,c='b',marker='_',s=60)

plt.show()








