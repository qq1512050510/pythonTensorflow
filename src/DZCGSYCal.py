#!/usr/bin/env python
# -*- coding: utf-8 -*-  @Author  : bjsasc
##
#电子常规试验方案算法

import xmltodict
import math
import string


def get_input_pare()->dict:
    with open('DZCGSYFile.syfa') as f:
        input_par = xmltodict.parse(f.read())

        k = input_par['电子常规试验方案']['InputDatas']
        par = k['InputData']
        r = {}
        for k in par:
            r[k['@Name']] = k['@value']
    return r


def process()->dict:
    result = dict()
    in_pare = get_input_pare()
    if int(in_pare['可靠性指标类型']) == 0:
        if int(in_pare['样本量类型']) == 0:
            result = CalculateKKDXX(in_pare)
            print('计算方法1')
        else:
            #result = CalculateKKDGS(in_pare)
            print('计算方法2')
    else:
        if int(in_pare['样本量类型']) == 0:
            result = CalculateKKDGS_DYBL(in_pare)
        else:
            result = CalculateKKDGS_DSYSJ(in_pare)

    return result


def out_put(result:dict):
    outdict = xmltodict.unparse(result)

    with open('result.xml', 'w+',encoding='utf-8') as f:
        f.write(outdict)
        f.flush()
        f.close()


def CalculateKKDXX(input:dict):
    d = dict()
    r = input['任务时间要求']
    d['result'] = '计算结果' + str(r)
    return d


def CalculateKKDGS_DYBL(input:dict):
    retdic = dict()
    iTaskTime = int(input['任务时间要求'])
    dblKKDDGJ = float(input['可靠度点估计'])
    iYXSXS = int(input['允许失效数'])
    dblLSSYXX = float(input['历史试验信息'])
    iLSSXS = int(input['历史失效数'])
    dblYXSYSH = float(input['允许试验时间'])
    dblDWSJCB = float(input['单位时间成本'])
    dblDWYPCB = float(input['单位样品成本'])
    iSDYBL= int(input['样本量'])
    iTemp = -1.0

    # 等效试验时间
    dblCalDXSYSJ = 0.0
    if (iYXSXS == 0 or dblLSSYXX == 0):
        dblCalDXSYSJ = ((math.log(0.5) / math.log(dblKKDDGJ, math.e)) * iTaskTime) - dblLSSYXX
    else:
        dblCalDXSYSJ = (iTemp * (((iYXSXS + iLSSXS) / math.log(dblKKDDGJ, math.e)) * iTaskTime)) - dblLSSYXX

    iDXSYSJ = int(dblCalDXSYSJ + 0.5)

    # 计算样本量
    dblYBL = math.sqrt((dblDWYPCB * dblCalDXSYSJ) / dblDWSJCB)
    iYBL = int(dblYBL + 0.5)

    # 计算的试验时间
    #dblSYSJ = iCalDXSYSJ / iYBL
    dblSYSJ = dblCalDXSYSJ / dblYBL
    iSYSJ = int(dblSYSJ + 0.5)

    # 试验总成本
    dblSYZCB = dblDWSJCB * dblYBL + dblDWYPCB * dblSYSJ
    #dblSYZCB = dblDWSJCB * iYBL + dblDWYPCB * iSYSJ
    iSYZCB = int(dblSYZCB + 0.5)

    #输出结果
    childdic = dict()
    childdic['result1'] = '可靠度点估计'
    childdic['result2'] = '定样本量'
    childdic['result3'] = '计算结果：'
    childdic['result4'] = '等效试验时间：' + str(iDXSYSJ)
    childdic['result5'] = '样本量：' + str(iYBL)
    childdic['result6'] = '单位试验样品的成本：' + str(int(dblDWSJCB))
    childdic['result7'] = '单位试验时间的成本：' + str(int(dblDWYPCB))
    childdic['result8'] = '最优化的试验样本量：' + str(iYBL)
    childdic['result9'] = '试验时间：' + str(iSYSJ)
    childdic['result10'] = '试验总成本：' + str(iSYZCB)

    retdic['result'] = childdic

    return retdic


def CalculateKKDGS_DSYSJ(input:dict):
    retdic = dict()
    iTaskTime = int(input['任务时间要求'])
    dblKKDDGJ = float(input['可靠度点估计'])
    iYXSXS = int(input['允许失效数'])
    dblLSSYXX = float(input['历史试验信息'])
    iLSSXS = int(input['历史失效数'])
    dblYXSYSH = float(input['允许试验时间'])
    dblDWSJCB = float(input['单位时间成本'])
    dblDWYPCB = float(input['单位样品成本'])
    dblSDSYSJ= float(input['允许试验时间'])
    iTemp = -1.0

    #等效试验时间
    dblCalDXSYSJ = 0.0
    if (iYXSXS == 0 or dblLSSYXX == 0):
        dblCalDXSYSJ = ((math.log(0.5) / math.log(dblKKDDGJ, math.e)) * iTaskTime) - dblLSSYXX
    else:
        dblCalDXSYSJ = (iTemp * (((iYXSXS + iLSSXS) / math.log(dblKKDDGJ, math.e)) * iTaskTime)) - dblLSSYXX

    iDXSYSJ = int(dblCalDXSYSJ + 0.5)

    #计算样本量
    dblYBL = math.sqrt((dblDWYPCB * dblCalDXSYSJ) / dblDWSJCB)
    iYBL = int(dblYBL + 0.5)

    #计算输出的样本量
    dblCalYBL = dblCalDXSYSJ / dblSDSYSJ
    iCalYBL = int(dblCalYBL + 0.5)

    #计算的试验时间
    #dblSYSJ = iCalDXSYSJ / iYBL
    dblSYSJ = dblCalDXSYSJ / dblYBL
    iSYSJ = int(dblSYSJ + 0.5)

    #试验总成本
    dblSYZCB = dblDWSJCB * dblYBL + dblDWYPCB * dblSYSJ
    #dblSYZCB = dblDWSJCB * iYBL + dblDWYPCB * iSYSJ
    iSYZCB = int(dblSYZCB + 0.5)

    # 输出结果
    childdic = dict()
    childdic['result1'] = '可靠度点估计'
    childdic['result2'] = '定试验时间'
    childdic['result3'] = '计算结果：'
    childdic['result4'] = '等效试验时间：' + str(iDXSYSJ)
    childdic['result5'] = '样本量：' + str(iCalYBL)
    childdic['result6'] = '单位试验样品的成本：' + str(int(dblDWSJCB))
    childdic['result7'] = '单位试验时间的成本：' + str(int(dblDWYPCB))
    childdic['result8'] = '最优化的试验样本量：' + str(iYBL)
    childdic['result9'] = '试验时间：' + str(iSYSJ)
    childdic['result10'] = '试验总成本：' + str(iSYZCB)

    retdic['result'] = childdic

    return retdic


if __name__ == '__main__':
    d = process()
    print(d)
    out_put(d)
