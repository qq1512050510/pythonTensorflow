#-*- coding:utf-8 -*-

"""
"""
import numpy as np
#import matplotlib.pyplot as plt
# compatibility mac
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
x_train = np.linspace(-1,1,101)
print(x_train)
y_train = 2*x_train + np.random.rand(*x_train.shape)*0.33
print(x_train.shape)
print(y_train)
print(*x_train.shape)
print(x_train.shape)
m = [1,2]
print(type(m))
plt.scatter(x_train,y_train)
plt.show()
