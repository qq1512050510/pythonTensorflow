#-*-coding:utf-8 -*-
'''
Created on 2018年10月30日

@author: adp
'''
# 可写函数说明
import math

def printinfo( arg1, *vartuple ):
    "打印任何传入的参数"
    print ("输出: ")
    print (arg1)
    for var in vartuple:
        print (var)
    return;

def funDestimate(r,rh,Rl,n,gama,t,th):
    
    return 1;
def funDestimateN(r,rh,Rl,n,gama,t,th):
    return 1;

def dEstimate(r,rh,Rl,n,gama,t,th):
    
    if r==0 and rh==0:
        result = funDestimate(r,rh,Rl,n,gama,t,th) 
    else:
        result = funDestimateN(r,rh,Rl,n,gama,t,th)
           
    return result;

print(dEstimate(0,0,1,1,1,1,1))