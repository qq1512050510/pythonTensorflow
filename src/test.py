# -*- coding:utf-8 -*-
'''
Created on 2018年7月28日

@author: adp
'''
import tensorflow as tf
import math
math.log
sess = tf.Session()
hello = tf.constant("hello tentensorflowTest")
print(sess.run(hello))

print ("你好")
if __name__ == '__main__':
    pass

Epsilon = 10e-16
ln10 = 2.30258509299404568401
"""
def ln_h(x):
    '''
    ln函数泰勒换元展开
    :param x: 0<x
    :return:ln(x)
    '''
    def ln_h1(x):
        s2 = 0.0
        delta = x = (x - 1.0) / (x + 1.0)
        i = 0
        while fab_h(delta * 2) / (i * 2 + 1) > Epsilon:
            s2 += delta / (i * 2 + 1)
            delta *= x * x
            i += 1
        print(i)
        return 2 * s2
    coef = 0
    if x > 10:
        while x / 10 > 1:
            coef += 1
            x /= 10
        return ln_h1(x) + coef*ln10
    elif x < 1:
        while x * 10 < 10:
            coef += 1
            x *= 10
        return ln_h1(x) - coef*ln10
    else:
        return ln_h1(x)
"""