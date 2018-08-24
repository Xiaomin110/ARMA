# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 10:28:35 2018

@author: hugen
"""
'''
序列平稳，不做任何处理，直接建模预测
'''

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import acf, pacf
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

####读取数据
data=pd.read_csv('C:/Users/hugen/Desktop/migu_data.csv')
df=data[['date','ZJ027171']]
df=df[18:]###从完整月份开始
#print(df.head(20))
####把字符串格式转化为日期格式
df.index=pd.to_datetime(df['date'],format='%Y%m%d')
ts=df['ZJ027171']###取出指标那列数据
ts=ts.astype('float64')
#print(ts)

####留取部分数据作为测试集
ts_test=ts[-9:-4]###测试
ts=ts[:-9]###训练集

print('训练集:\n',ts.tail())
print('测试集:\n',ts_test)

#########################################################################1
######################################数据序列白噪声检验
ts_lg = acorr_ljungbox(ts, lags=1)###原假设：白噪声
print(ts_lg[1])#<0.05,数据非白噪声,有建模价值

#####################################平稳性的统计检验模块：
###时序图(看数据的波动趋势的周期性)
def draw_ts(timeSeries):
    f = plt.figure(facecolor='white')
    timeSeries.plot(color='blue')
    plt.show()
####ADF检验
def ADF_test(timeseries):###原假设：非平稳
    print('Results of Dickey-Fuller Test:')
    df_test = adfuller(timeseries, autolag='AIC')# df_test的输出前一项依次为检测值，p值，滞后数，使用的观测数，各个置信度下的临界值
    df_output = pd.Series(df_test[0:4], index=['Test Statistic', 'p-value', 
                          '#Lags Used', 'Number of Observations Used'])
    for key, value in df_test[4].items():
        df_output['Critical value (%s)' % key] = value
        return df_output
print('原数据波动情况：',draw_ts(ts))
print('原数据平稳性：',ADF_test(ts))###<0.05,平稳


#########################################################################3
##################################模型定阶

def proper_model(data_ts, maxLag):
    init_bic = float("inf")
    init_p = 0
    init_q = 0
    init_properModel = None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            try:
                model = ARMA(data_ts, order=(p, q))
                results_ARMA = model.fit(disp=-1, method='css')
            except:
                continue
            bic = results_ARMA.bic
            if bic < init_bic:
                init_p = p
                init_q = q
                init_properModel = results_ARMA
                init_bic = bic
    print(u'BIC最小值%s的p值、q值为：%s,%s'%(init_bic,init_p,init_q))
    return init_bic,init_p, init_q
maxlag=3
'''
for lag in range(1,maxlag):
   proper_model(tr_diff,lag)  ##(3,3)
'''
#######################################################################4
###############################模型训练
order=proper_model(ts,maxlag)
model = ARMA(ts, order=(order[1],order[2]))

###输出模型形式
result_arma = model.fit(disp=-1, method='css')
print('模型信息报告：\n',result_arma.summary2())
#print('forecast:',result_arma.forecast(5))
#summary = (result_arma.summary2(alpha=.05, float_format="%.8f"))
#print(summary)

###############################模型检验
### 残差 ADF 检验
model_resid =result_arma.resid
adf_test=st.adfuller(model_resid)
resid_adf_output=pd.DataFrame(index=["Test Statistic Value", "p-value", "Lags Used", 
                                     "Number of Observations Used","Critical Value(1%)",
                                     "Critical Value(5%)","Critical Value(10%)"],columns=['value'])
resid_adf_output['value']['Test Statistic Value'] = adf_test[0]
resid_adf_output['value']['p-value'] = adf_test[1]
resid_adf_output['value']['Lags Used'] = adf_test[2]
resid_adf_output['value']['Number of Observations Used'] = adf_test[3]
resid_adf_output['value']['Critical Value(1%)'] = adf_test[4]['1%']
resid_adf_output['value']['Critical Value(5%)'] = adf_test[4]['5%']
resid_adf_output['value']['Critical Value(10%)'] = adf_test[4]['10%']
print('残差ADF检验:\n',resid_adf_output)

####残差白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox
lg = acorr_ljungbox(model_resid, lags=1)###原假设：白噪声
print('残差白噪声检验:\n',lg[1])#>0.05残差是白噪声序列,模型信息提取较充分

##########################################################################5
################################模型拟合评价
#####模型预测值
pred_ts = result_arma.predict()
pred_ts.dropna(inplace=True)
print(pred_ts.tail())
###########模型评价

###绘图比较原数据和拟合数据
ts_1 = ts[pred_ts.index]  # 过滤没有预测的记录
plt.figure(facecolor='white')
pred_ts.plot(color='blue', label='Predict')
ts_1.plot(color='red', label='Original')
plt.legend(loc='best')
plt.title('RMSE: %.4f'% np.sqrt(sum((pred_ts-ts_1)**2)/ts_1.size))
plt.show()

######统计指标评价模型
abs_=(pred_ts-ts_1).abs()
mae_=abs_.mean()
rmse_=((abs_**2).mean())**0.5
mape_=(abs_/ts_1).mean()
print('平均绝对误差为：%0.4f,\n均方根误差：%0.4f,\n平均绝对百分误差：%0.6f。' %(mae_,rmse_,mape_))

###########################################################################6
###############################模型样本外预测

#####外推预测期数和预测值
step_n=5
df_forecast=result_arma.forecast(steps=step_n,alpha=0.05) 

#############样本外预测的置信区间
df=pd.concat([pd.DataFrame(df_forecast[0], columns=['模型预测值']),
              pd.DataFrame(df_forecast[1], columns=['预测标准差']),
              pd.DataFrame(df_forecast[2], columns=['区间估计下限(95%)','区间估计上限(95%)'])],axis=1)
print('样本外预测的置信区间：',df)

######生成与测试集数据ts_test相同的索引列

idx = pd.date_range(start=ts_test.index[0],periods=step_n, freq='D')
######索引值起点为测试集索引起点ts_test.index[0]，索引长度为预测期数（测试集长度）step_n
fc = pd.Series(df_forecast[0],index=idx)
l_95 = pd.Series(df_forecast[2][:,0],index=idx)
u_95 = pd.Series(df_forecast[2][:,1],index=idx)
print(idx)
print('fc:\n',fc)
#print('l_95:\n',l_95)
#print('u_95:\n',u_95)
#l_95=l_95.reset_index(drop = True)
#fc=fc[ts_test.index]

########样本外预测值与测试集数据的比较分析

model_fore=pd.concat([pd.DataFrame(ts_test),pd.DataFrame(fc),pd.DataFrame(l_95),
                      pd.DataFrame(u_95)],axis=1,ignore_index=True)
model_fore[4]=model_fore[1]-model_fore[0]
model_fore[5]=model_fore[2]-model_fore[0]
model_fore[6]=model_fore[3]-model_fore[0]
model_fore[7]=model_fore[3]-model_fore[2]
model_fore.columns=['待测数据','模型预测值','预测区间lower值','预测区间upper值',
                    '预测误差','预测下限差值','预测上限差值','预测区间长度']
print('模型预测信息',model_fore)    

abs_=(fc-ts_test).abs()
mae_=abs_.mean()
rmse_=((abs_**2).mean())**0.5
mape_=(abs_/ts_test).mean()
print('平均绝对误差为：%0.4f,\n均方根误差：%0.4f,\n平均绝对百分误差：%0.6f。' %(mae_,rmse_,mape_))


plt.subplot(212)
fc.plot(color='blue', label='Predict') # 预测值
ts_test.plot(color='red', label='Original') # 真实值
l_95.plot(color='grey', label='low') # 低置信区间
u_95.plot(color='grey', label='high') # 高置信区间
plt.legend(loc='best')
plt.title('RMSE: %.4f' % np.sqrt(sum((fc-ts_test) ** 2) / ts_test.size))
plt.tight_layout()
plt.show()


###################################异常判断

def outlier_report(ts_test,diff_recover1,diff_recover2,diff_recover3):
    #####输入的必须是有相同索引的series
    
    data_fore=pd.concat([pd.DataFrame(ts_test),pd.DataFrame(diff_recover1),pd.DataFrame(diff_recover2),
                         pd.DataFrame(diff_recover3)],axis=1,ignore_index=True)
    #print(data_fore.ix[0,1])####取出第一行第二列的值
    #print(data_fore[0][1])####取出第一列第二行的值

    for i in range(0,len(data_fore)):
        if (data_fore.ix[i,0]<data_fore.ix[i,2])|(data_fore.ix[i,0]>data_fore.ix[i,3]):
            data_fore.ix[i,4]='异常'
        else:
            data_fore.ix[i,4]='正常'
    data_fore.columns=['待测数据','模型预测值','预测区间lower值','预测区间upper值','指标状态']
    print('指标状态评估报告：\n',data_fore)
    return data_fore

outlier_report(ts_test,fc,l_95,u_95)
#print(data_fore) 
#data_fore.to_csv('C:/Users/hugen/Desktop/data_fore2.csv')

