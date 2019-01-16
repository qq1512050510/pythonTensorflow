#-*-coding:utf-8 -*-
'''
Created on 2018年10月29日

@author: adp
'''

# -*- coding: utf-8 -*-
# Toby QQ：231469242
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import math,pylab,matplotlib,numpy
from matplotlib.font_manager import FontProperties
 
#设置中文字体
font=FontProperties(fname=r"c:\windows\fonts\simsun.ttc",size=15)
 
n=10
 
#绘制自由度为n的卡方分布图,n表示生成卡方数组的个数
def Get_chisquareDatas(n):
    #标准正太分布
    normalDistribution=stats.norm(0,1)
    list_data=[]
    for i in range(n):
        normal_data=normalDistribution.rvs(30)
        chisquare_data=normal_data**2
        list_data.append(chisquare_data)
    return list_data
     
def Plot_chisquare(n):
    list_data=Get_chisquareDatas(n)
    sum_data=sum(list_data)
    plt.hist(sum_data)

print('123')
Plot_chisquare(2)
Plot_chisquare(3)
Plot_chisquare(10)
print('123')