#coding:utf-8
import itertools
import numpy as np
import pandas as pd
#import cx_Oracle
from datetime import datetime,timedelta
import matplotlib.pylab as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.stattools as st
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt,ceil
import time
import sys
import warnings
import statsmodels.api as sm

#导入数据
def input_data(datafile):
    data = pd.read_excel(datafile,index_col='date')
    #data.index.name=None
    data.index = pd.to_datetime(data.index,format='%Y%m%d')
    return data['ZJ027241']###选取对应指标数据

def data_smooth(data):
    dif = data.diff().dropna()
    td = dif.describe()#描述性统计数据得到：min,25%,50%,75%,max值
    high = td['75%']+1.5*(td['75%']-td['25%'])#定义高点阈值，1.5倍四分位距之外
    low = td['25%']-1.5*(td['75%']-td['25%'])#定义低点阈值
    #####   high: 607124.25,low: 43014.25 ###################################################

    forbid_index = dif[(dif > high)|(dif < low)].index
    print(forbid_index)
    i = 0
    while i < len(forbid_index) - 1:
        n = 1#发现连续多少个点变化幅度过大，大部分只有单个点
        start = forbid_index[i]#异常点的起始索引|2017-12-01
        end = start + timedelta(days=n)
        while end in forbid_index:###判断上结束时间点是否在索引列表内
            end = end + timedelta(days=n)#2017-12-03
        #start = pd.Timestamp(start).strftime('%Y-%m-%d')###转换格式
        #end = pd.Timestamp(end).strftime('%Y-%m-%d')
        #value = data.truncate(before='2017-11-01', after='2017-11-14')
        #value = data.truncate(before=start - timedelta(days=5),after=start + timedelta(days=1)))
        value = np.linspace(data[start - timedelta(days=1)], data[end],3)
        data[start] = value[1]
        i += 1
    return data


if __name__ == '__main__':
    datafile = 'data2.xls'
    data = input_data(datafile)
    data = data[3:]
    #data = data.astype('float64')
    #data.plot()
    #plt.show()
    #data = data_smooth_all(data)
    #input('>>>>>>>>>>>>>')
    #data = data_smooth(data)
    data.plot()
    plt.show()



