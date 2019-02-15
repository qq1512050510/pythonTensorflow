'''
Created on 2019年2月14日

@author: jyp
'''
#matplotlib inline
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (14,8)

n_observations = 100
xs = np.linspace(-3,3,n_observations)
print(type(xs))

