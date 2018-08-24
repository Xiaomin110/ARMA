# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 10:06:05 2018

@author: hugen
"""
'''
原始序列不平稳，经过分解和差分后，建模
'''

import warnings
warnings.filterwarnings("ignore") ####忽略警告

import numpy as np
np.set_printoptions(suppress=True)####改变科学计数形式（没用）

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
df=data[['date','ZJ027281']]
#print(df)

df=df[18:]###从完整月份开始
#print(df.head(20))
####把字符串格式转化为日期格式
df.index=pd.to_datetime(df['date'],format='%Y%m%d')
ts=df['ZJ027281']###取出指标那列数据
ts=ts.astype('float64')
#print(ts)

####留取部分数据作为测试集
ts_test=ts[-2:]###测试
ts=ts[:-2]###训练集

print('训练集:',ts.tail())
print('测试集:',ts_test)


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


########################################################################2
##############======================平稳性处理模块：

####分解(decomposing)
def decompose(timeseries):
    # 返回包含三个部分 trend（趋势部分）， seasonal（季节性部分） 和residual (残留部分)
    decomposition = seasonal_decompose(timeseries,model="additive",two_sided=False)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.subplot(411)
    plt.plot(ts, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()

    return trend, seasonal, residual

trend, seasonal, residual=decompose(ts)
#print('trend的平稳性:\n',ADF_test(trend.dropna()))
#print('seasonal的平稳性:\n',ADF_test(seasonal.dropna()))
#print('residual的平稳性:\n',ADF_test(residual.dropna()))

##################剔除季节性成分
#####留下趋势trend和残差residual部分进行处理，建模。
tr=ts-seasonal
print('tr的平稳性:\n',ADF_test(tr.dropna()))###>0.05,非平稳

#################################################tr差分

def diff_ts(ts, d):
    global shift_ts_list
    #  动态预测第二日的值时所需要的差分序列
    global last_data_shift_list
    shift_ts_list = []
    last_data_shift_list = []
    tmp_ts = ts
    for i in d:
        last_data_shift_list.append(tmp_ts[-i])
        shift_ts = tmp_ts.shift(i)
        shift_ts_list.append(shift_ts)
        tmp_ts = tmp_ts - shift_ts
    tmp_ts.dropna(inplace=True)
    return tmp_ts

tr=tr.dropna()###去掉前面几个空值
tr_diff=diff_ts(tr, [1])
tr_diff.dropna(inplace=True)###返回drop的部分（none）
print('tr差分后的平稳性:\n',ADF_test(tr_diff))###<0.05,平稳

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
order=proper_model(tr_diff,maxlag)
model = ARMA(tr_diff, order=(order[1],order[2]))

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
pred_tr_diff = result_arma.predict()
pred_tr_diff.dropna(inplace=True)
#rss=sum((pred_ts_trend_diff-ts_trend_diff)**2)
#print(rss)

############预测值还原
# 一阶差分还原
tr_diff_shift =tr_diff.shift(1)###趋势trend的一阶差分值（模型入参数数据）前移一位
tr_diff_recover= pred_tr_diff.add(tr_diff_shift)
###趋势还原
tr_shift=tr.shift(1)
tr_recover=tr_diff_recover.add(tr_shift)
###分解还原
ts_recover=tr_recover+seasonal
ts_recover.dropna(inplace=True)###去空值

###########模型评价

###绘图比较原数据和拟合数据
ts_1 = ts[ts_recover.index]  # 过滤没有预测的记录
plt.figure(facecolor='white')
ts_recover.plot(color='blue', label='Predict')
ts_1.plot(color='red', label='Original')
plt.legend(loc='best')
plt.title('RMSE: %.4f'% np.sqrt(sum((ts_recover-ts_1)**2)/ts_1.size))
plt.show()

######统计指标评价模型
print('训练数据的预测还原后：\n', ts_recover)
print('训练数据\n', ts_1)
print('训练误差率：\n', (ts_recover-ts_1).abs()/ts_1*100)
    
abs_=(ts_recover-ts_1).abs()
mae_=abs_.mean()
rmse_=((abs_**2).mean())**0.5
mape_=(abs_/ts_1).mean()
print('训练数据: \n平均绝对误差为：%0.4f,\n均方根误差：%0.4f,\n平均绝对百分误差：%0.6f。' %(mae_,rmse_,mape_))

################################ ###########################################6
###############################模型样本外预测
##########样本外预测（非平稳序列）

###### 差分还原函数
def  predict_diff_recover(predict_value, d):
    if isinstance(predict_value, float):
        tmp_data = predict_value
        for i in range(len(d)):
            tmp_data = tmp_data + last_data_shift_list[-i-1]
    elif isinstance(predict_value, np.ndarray):
        tmp_data = predict_value[0]####选第一个值
        for i in range(len(d)):
            tmp_data = tmp_data + last_data_shift_list[-i-1]
    else:
        tmp_data = predict_value
        for i in range(len(d)):
            tmp_data = tmp_data.add(shift_ts_list[-i-1])  #从后往前一步一步还原
    #print('tmp_data',tmp_data)
    return tmp_data    

#########样本外预测函数
def out_sample_pre(period,seasonal,result_arma,diff_shift_ts,d):
    ####diff_shift_ts是差分还原后的数据，这里是tr_recover
    #print('输入:',diff_shift_ts)
    
    output = result_arma.forecast(steps=period,alpha=0.1)###显著性水平0.1,置信水平越高99%，置信区间越大，数据预测精度相对低，95%的可能性落入
    print('置信水平为：95%的置信区间：',output[2])
    ##print(u'置信水平为：%s的置信区间为：%s' %((1-alpha),output[2]))
    
    fc= output[0]
    l_u95 = output[2]
    l95 = l_u95[:,0]
    u95=l_u95[:,1]
    print('l95:',l95)
    print('u95:',u95)
    
    #样本外差分预测
    
    idx = pd.date_range(seasonal.index[-1], periods=period+1, freq='D',closed='right')
    seasonal_id = []
    sea=[]   
    for id in seasonal:
        if id not in seasonal_id:
            seasonal_id.append(id)
    p=seasonal_id.index(seasonal[-1])
    new_seasonal= seasonal_id * int(period / len(seasonal_id)+2)
    for i in range(period):
           sea.append(new_seasonal[p+i+1]) #第一个元素
    sea1 = pd.Series(sea,index=idx)
    seasonal1=seasonal.append(sea1)
    
    #样本外差分还原
    pred1=[]
    l_95=[]
    u_95=[]
    #一阶差分，多步预测
    #多步预测使用最后一个的预测值
    k=d[0]
    for i in range(k):
        pred1.append(fc[i]+diff_shift_ts[i-k])   
        l_95.append(l95[i]+diff_shift_ts[i-k])   
        u_95.append(u95[i]+diff_shift_ts[i-k])   
    for i in range(period-k):
        pred1.append(fc[i+k]+pred1[i])  
        l_95.append(l95[i+k]+l_95[i])   
        u_95.append(u95[i+k]+u_95[i])    
        '''
    print('l_95:',l_95)
    print('u_95:',u_95)
    print('l95:',l95)
    print('u95:',u95)
    '''
    #多阶差分，一步预测
    #一步预测使用最后一个的真实值
    pred1= predict_diff_recover(fc,d)####用第一个值进行还原
    l_95 =predict_diff_recover(l95,d)  
    u_95 =predict_diff_recover(u95,d)
    '''
    print('pred1:',pred1)
    print('l_95:',l_95)
    print('u_95:',u_95)
    '''
    yhat = pd.Series(pred1,index=idx)####一个复制成多个
    l_hat= pd.Series(l_95,index=idx)
    u_hat= pd.Series(u_95,index=idx)
    '''
    print('yhat:',yhat)
    print('l_hat:',yhat)
    print('h_hat:',yhat)
    '''
    
    diff_shift_ts1=diff_shift_ts.append(yhat)#####多个预测的趋势还原
    diff_shift_ts2=diff_shift_ts.append(l_hat)
    diff_shift_ts3=diff_shift_ts.append(u_hat)
    '''
    print('diff_shift_ts1:',diff_shift_ts1.tail())
    print('diff_shift_ts2:',diff_shift_ts2.tail())
    print('diff_shift_ts3:',diff_shift_ts3.tail())
    '''
    diff_recover1=diff_shift_ts1+seasonal1###多个预测的分解还原
    diff_recover2=diff_shift_ts2+seasonal1
    diff_recover3=diff_shift_ts3+seasonal1
   # print('diff_recover1:',diff_recover1.tail())
    
    diff_recover1.dropna(inplace=True)
    diff_recover2.dropna(inplace=True)
    diff_recover3.dropna(inplace=True)
    return diff_recover1,diff_recover2,diff_recover3

diff_recover1,diff_recover2,diff_recover3=out_sample_pre(5,seasonal,result_arma,tr_recover,[1])

print('预测值y:\n',diff_recover1.tail(5))
print('预测值l:\n',diff_recover2.tail(5))
print('预测值u:\n',diff_recover3.tail(5))
print('预测区间长度d:\n',(diff_recover2-diff_recover3).abs().tail(5))

########样本外预测值与测试集数据的比较分析
####预测误差
ts_test.dropna(inplace=True)
diff_recover1=diff_recover1[ts_test.index]
diff_recover2=diff_recover2[ts_test.index]
diff_recover3=diff_recover3[ts_test.index]
print('测试集值：\n',ts_test)
print('预测差值:\n',(diff_recover1-ts_test))
print('预测下限差值:\n',(diff_recover2-ts_test))
print('预测上限差值:\n',(diff_recover3-ts_test))

####误差指标分析
abs_=(diff_recover1-ts_test).abs()
mae_=abs_.mean()
rmse_=((abs_**2).mean())**0.5
mape_=(abs_/ts_test).mean()
print('测试数据：\n平均绝对误差为：%0.4f,\n均方根误差：%0.4f,\n平均绝对百分误差：%0.6f。' %(mae_,rmse_,mape_))



####绘图比较
plt.subplot(212)
diff_recover1.plot(color='blue', label='Predict') # 预测值
ts_test.plot(color='red', label='Original') # 真实值
diff_recover2.plot(color='black', label='low') # 低置信区间
diff_recover3.plot(color='black', label='high') # 高置信区间
plt.legend(loc='best')
plt.title('RMSE: %.4f' % np.sqrt(sum((diff_recover1 - ts_test) ** 2) / ts_test.size))
plt.tight_layout()
plt.show()

###################################异常判断

def outlier_report(ts_test,diff_recover1,diff_recover2,diff_recover3):
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

data_fore=outlier_report(ts_test,diff_recover1,diff_recover2,diff_recover3)
#print(data_fore)
#data_fore.to_csv('C:/Users/hugen/Desktop/data_fore2.csv')
