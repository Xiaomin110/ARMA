#coding:utf-8
import numpy as np
import pandas as pd
#import cx_Oracle
from datetime import datetime,timedelta
import matplotlib.pylab as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.tsa.api as smt
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.stattools as st
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt,ceil
import time

np.set_printoptions(suppress=True)##设置代码中输出的数字不以科学计数发输出

#导入数据
def input_data(datafile):
    data = pd.read_excel(datafile,index_col='date')
    #data.index.name=None
    data.index = pd.to_datetime(data.index,format='%Y%m%d')
    return data['ZJ027161']###选取对应指标数据

#平稳性检验
def adf_test(ts):
    adf = adfuller(ts,autolag='AIC')
    return adf

#白噪声检验
def acorr_test(ts):
    lg=acorr_ljungbox(ts,lags=1)
    return lg

#最优差分
def diff_test(ts):
    '''
    :param ts: 原始序列
    :return: adf值，差分次数，差分后平稳序列,经过差分的全部序列
    '''
    adf = adf_test(ts)
    diff_num = 0
    ts_diff = ts
    global ts_diff_all
    ts_diff_all = []
    while adf[1] >= 0.05:
        diff_num = diff_num+1
        ts_diff = ts.diff(diff_num).dropna()
        ts_diff_all.append(ts_diff)#添加到差分数据列表中
        adf = adf_test(ts_diff)
    print(ts_diff)
    print(u'原始序列经过%s阶差分后归于平稳，p值为%s' %(diff_num,adf[1]))
    return adf,diff_num,ts_diff,ts_diff_all

#非平稳序列差分预处理
def data_prepro(ts):
    '''
    :param ts: 原始非平稳序列(带有季节性数据)
    :return:差分后的平稳序列值，差分次数，和季节性部分数据
    '''
    ts_log = np.log(ts)  #取对数--针对原数据值域比较大的，为了缩小值域，采用对数化。
    decomposition = seasonal_decompose(ts_log, model="additive",two_sided=False)#把时序数据分解成下面三部分
    trend = decomposition.trend#趋势部分
    seasonal = decomposition.seasonal#季节部分
    residual = decomposition.resid#残余部分

    '''
    #数据展示
    plt.subplot(411)
    plt.plot(ts_log, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    '''
    adf_tre = adfuller(trend.dropna(),autolag='AIC')#非平稳
    adf_season = adfuller(seasonal.dropna(),autolag='AIC')#平稳
    adf_resid = adfuller(residual.dropna(),autolag='AIC')#平稳
    return trend,seasonal,residual


#模型定阶
def model_test(ts,d,maxlag):
    '''
    :param data_ts: 平稳序列
    :param maxLag:p,q最大阶数
    :return:bic值，p,q值，训练的模型
    '''
    init_bic = float("inf")#正无穷大
    init_p = 0
    init_q = 0
    init_properModel = None
    for p in np.arange(maxlag):
        print(p)
        for q in np.arange(maxlag):
            print(q)
            try:
                model = ARMA(ts, order=(p, q))
                results_MODEL = model.fit(disp=-1, method='css')#不输出信息，’css‘方法为条件平方和
            except:
                continue
            bic = results_MODEL.bic#获取到该p,q数值下模型的bic值的大小
            if bic < init_bic:
                init_p = p
                init_q = q
                init_properModel = results_MODEL
                init_bic = bic
    print(u'BIC最小值%s的p值、d值和q值为：%s,%s,%s'%(init_bic,init_p,d,init_q))
    return init_p, init_q,init_properModel

#差分还原
def predict_diff_recover(predict_value,ts_diff_all, d):
    '''
    :param predict_value: 预测值
    :param ts_diff: 差分数组 [ts_diff_1,ts_diff_2,ts_diff_3,ts_diff_4...]
    :param d: 差分次数
    :return:还原好的预测数据
    '''
    if d > 0:
        #一阶一阶差分还原
        diff_recover = predict_value
        for i in range(1, d+1):
            i = -i
            diff_shift_ts = ts_diff_all[i].shift(1)
            diff_recover = diff_recover.add(diff_shift_ts)
            '''
            #二阶差分还原
            diff_shift_ts = ts_diff[2].shift(1)
            diff_recover = diff_recover.add(diff_shift_ts)
            '''
        return diff_recover
    else:#如果数据没有差分
        print(u'该数据没有进行差分')
        return None

##样本外预测
def predict_new(adf_tre, seasonal, n, model, data, log_recover_tre, d):
    '''
    :param adf-tre:趋势性数据序列
    :param seasonal: 季节性数据序列
    :param n: 往后预测的个数
    :param model: 训练模型
    :param ts_diff_tre_all: 还原到
    :param d:差分次数
    :return: 预测数据
    '''
    output = model.forecast(steps=n, alpha=0.1)#步长和置信区间，返回预测值，标准差，置信区间
    fc= output[0]#预测值
    l_u95 = output[2]#置信区间
    l95 = l_u95[:, 0]#下限
    u95=l_u95[:, 1]#上限

    pred_time_index = pd.date_range(start=seasonal.index[-1], periods=n+1, freq='1D')[1:]#预测数据索引
    ###############################季节性处理############################
    seasonal_id = []
    sea=[]
    print('seasonal:', seasonal)
    for id in seasonal:
        if id not in seasonal_id:
            seasonal_id.append(id)###获取一个周期内的数据列表
    print('seasonal_id:',seasonal_id)
    p=seasonal_id.index(seasonal[-1])#获取索引，判断已有数据中最后一个属于周期列表的第几个值。p=0(第一个),则预测第一个数据对应周期性为第二个
    new_seasonal= seasonal_id * int(n / len(seasonal_id)+2)###将上述周期性列表扩充成2个周期性
    for i in range(n):##上面的扩充处理防止n大于周期列表长度
           sea.append(new_seasonal[p+i+1]) #第二个元素
    sea1 = pd.Series(sea,index=pred_time_index)
    seasonal1=seasonal.append(sea1)##原始周期性数据加上了后面需要预测的几个周期数据
    ###样本外差分还原
###对趋势预测数据的进行还原######
    pred1 = []
    l_95 = []
    u_95 = []
    if n>1:#多步预测
        if d>0:#原始序列进行了差分
            #if isinstance(fc,np.ndarray):
            for i in range(d):
                pred1.append(fc[i]+log_recover_tre[i-d])
                l_95.append(l95[i]+log_recover_tre[i-d])
                u_95.append(u95[i]+log_recover_tre[i-d])
            for i in range(n-d):
                pred1.append(fc[i+d]+pred1[i])
                l_95.append(l95[i+d]+l_95[i])
                u_95.append(u95[i+d]+u_95[i])
        else:##没有进行差分处理的（无差分不需要加值还原，下面所诉是经过差分后的还原方法）
            ####有趋势性的原始序列，进行循环累加。
            #####先用最后一个真值加上趋势的预测值还原成真值预测值###############
            '''
            pred1.append(fc[0]+log_recover_tre[-1])
            l_95.append(l95[0]+log_recover_tre[-1])
            u_95.append(u95[0]+log_recover_tre[-1])
            #####循环相加赋值，就是用第一个真值预测值，去还原第二个真值预测值。
            for i in range(n-1):
                pred1.append(fc[i+1]+pred1[i])
                l_95.append(l95[i+1]+l_95[i])
                u_95.append(u95[i+1]+u_95[i])
            '''
            ###无趋势性数据，用样本最后一个值进行求和还原(无差分不需要加值还原)
            for i in range(n):
                pred1.append(fc[i])
                l_95.append(l95[i])
                u_95.append(u95[i])



    else:#一步预测
        if isinstance(fc,float):
            pred1 = fc + log_recover_tre[-1]
            l_95 = l95 + log_recover_tre[-1]
            u_95 = u95 + log_recover_tre[-1]
        '''
        pred1= predict_diff_recover(fc,ts_diff_tre_all,d)
        l_95 =predict_diff_recover(l95,ts_diff_tre_all,d)
        u_95 =predict_diff_recover(u95,ts_diff_tre_all,d)
        '''
    print('pred1:', pred1)
    print('l_95:', l_95)
    print('u_95:', u_95)
    #input('>>>>>>>>>>>>>>>>>stop')
        ####转换成时间序列数据，（相同的数据赋值成预测的个数）
    yhat = pd.Series(pred1,index=pred_time_index)
    l_hat= pd.Series(l_95,index=pred_time_index)
    u_hat= pd.Series(u_95,index=pred_time_index)
        ####附加在原始序列的列表末尾。增加20160101-20160105的数据到样本数据列表中
    log_recover_tre1 = log_recover_tre.append(yhat)
    log_recover_tre2 = log_recover_tre.append(l_hat)
    log_recover_tre3 = log_recover_tre.append(u_hat)
    print('log_recover_tre1',log_recover_tre1)
        ###加上前面处理好的季节性数据。
    log_recover_res1 = log_recover_tre1+ seasonal1
    log_recover_res2 = log_recover_tre2+ seasonal1
    log_recover_res3 = log_recover_tre3+ seasonal1
        ###对数还原
    log_recover1 = np.exp(log_recover_res1)
    log_recover2 = np.exp(log_recover_res2)
    log_recover3= np.exp(log_recover_res3)
    log_recover1.dropna(inplace=True)
    log_recover2.dropna(inplace=True)
    log_recover3.dropna(inplace=True)
    return log_recover1,log_recover2,log_recover3

##数据平滑
def data_smooth(data):
    data_index = data.index#平滑数据的索引
    dif = data.diff().dropna()
    td = dif.describe()#描述性统计数据得到：min,25%,50%,75%,max值
    high = td['75%']+3*(td['75%']-td['25%'])#定义高点阈值，3倍四分位距之外
    low = td['25%']-3*(td['75%']-td['25%'])#定义低点阈值
    #####   high: 607124.25,low: 43014.25 ###################################################

    forbid_index = dif[(dif > high)|(dif < low)].index
    print(forbid_index)
    i = 0
    while i < len(forbid_index):##末尾跳出循环
        n = 1#发现连续多少个点变化幅度过大，大部分只有单个点
        start = forbid_index[i] - timedelta(days=n)#异常点的起始索引|2017-12-01
        end = forbid_index[i] + timedelta(days=n)
        while end in forbid_index:###判断上结束时间点是否在索引列表内
            end = end + timedelta(days=n)#2017-12-03
            while end > data_index[-1]:##如果end加了天数之后大于最后一个值，则从前面取值
                end = start
                start = start - timedelta(days=n)
        #start = pd.Timestamp(start).strftime('%Y-%m-%d')###转换格式
        #end = pd.Timestamp(end).strftime('%Y-%m-%d')
        #value = data.truncate(before='2017-11-01', after='2017-11-14')
        if forbid_index[i] == data_index[0]:###突变点在原始数据列的第一位
            start = end##就是不在突变集合里面的数据
            end = end + timedelta(days=n)
            while end in forbid_index:###判断上结束时间点是否在索引列表内
                end = end + timedelta(days=n)#2017-12-03
        elif forbid_index[i] == data_index[-1]:####突变点在原始序列的最后一位
            ###取前面的处理好的数据进行平滑
            end = start
            start = end - timedelta(days=n)
        else:
            start = start
            end = end
        print('start',start)
        print('end',end)
        value = np.linspace(data[start], data[end],3)##求两个值的平均值
        print(value)
        data[forbid_index[i]] = value[1]
        i += 1
    return data

if __name__=='__main__':
    datafile = 'data2.xls'
    data = input_data(datafile)
    data = data[3:]###取出日期的连续数据（前两个日期数据中断）
    data_train = data[:-5]
    #data_train = data_smooth(data_train)
    data_test = data[-5:]
    #data.info()
    #data.index = pd.to_datetime(data, format='%Y%m%d')
    adf_tre, adf_season, adf_resid = data_prepro(data_train)#原始序列数据分解
    adf_tre = adf_tre + adf_resid###剔除季节性数据的结果
    #print('adf_tre序列值为%s'%adf_tre)
    #print('adf_season序列值为%s'%adf_season)
    #print('adf_resid序列值为%s'%adf_resid)

    #非平稳趋势性数据序列
    adf_tre_adf,adf_tre_num,adf_tre_diff,ts_diff_tre_all = diff_test(adf_tre.dropna())#非平稳的趋势性数据差分
    p_tre,q_tre,model_tre = model_test(adf_tre_diff,adf_tre_num,8)#返回定阶数和模型
    model_tre = ARMA(adf_tre_diff,order=(p_tre,q_tre))
    model_tre = model_tre.fit(disp=-1, method='css')
    print(model_tre.summary())


    rdtest = acorr_test(model_tre.resid)#模型残差白噪声检验
    print('rdtest[1]',rdtest[1])
    if rdtest[1] > 0.05:#残差为白噪声

        '''
        #平稳季节性数据序列
        adf_season_adf,adf_season_num,adf_season_diff,ts_diff_season_all = diff_test(adf_season.dropna())#非平稳的趋势性数据差分
        #p_season,q_season,model_season = model_test(adf_season_diff,adf_season_num,8)
        model_season = ARMA(adf_season_diff,order=(7,0))
        model_season = model_season.fit(disp=-1, method='css')

        #平稳残差数据序列
        adf_resid_adf,adf_resid_num,adf_resid_diff,ts_diff_resid_all = diff_test(adf_resid.dropna())#非平稳的趋势性数据差分
        #p_resid,q_resid,model_resid = model_test(adf_resid_diff,adf_resid_num,8)#返回模型阶数，和模型
        model_resid = ARMA(adf_resid_diff,order=(0,5))
        model_resid = model_resid.fit(disp=-1, method='css')
        '''
        predict_tre = model_tre.predict()#趋势序列模型预测
        #print('predict_tre序列值为:'+'\n',predict_tre)
        log_recover_tre = predict_diff_recover(predict_tre,ts_diff_tre_all,adf_tre_num)#数据差分还原

        #原始数据差分还原(先判断是否差分，在考虑是否还原)
        if log_recover_tre:
            diff_shift_ts = (adf_tre.dropna()).shift(1)
            log_recover_tre = log_recover_tre.add(diff_shift_ts)
        else:
            log_recover_tre = predict_tre.dropna()


        #print('log_recover_tre序列值为'+'\n',log_recover_tre)
        '''
        predict_season = model_season.predict()#季节模型预测
        print('predict_season序列值为%s'%predict_season)
        log_recover_season = predict_diff_recover(predict_season,ts_diff_tre_all,adf_tre_num)#数据还原

        diff_shift_ts = (adf_season.dropna()).shift(1)
        log_recover_season = log_recover_season.add(diff_shift_ts)
        print('log_recover_season序列值为%s'%log_recover_season)

        predict_resid = model_resid.predict()#残差模型预测
        print('predict_resid序列值为%s'%predict_resid)
        log_recover_resid = predict_diff_recover(predict_resid,ts_diff_tre_all,adf_tre_num)#数据还原

        diff_shift_ts = (adf_resid.dropna()).shift(1)
        log_recover_season = log_recover_resid.add(diff_shift_ts)
        print('log_recover_resid序列值为%s'%log_recover_resid)
        '''

        #result = log_recover_tre + log_recover_season + log_recover_resid#数据分解加法模型还原
        result = log_recover_tre + adf_season
        #print('result结果序列值为%s'%result)
        #对数还原
        log_recover = np.exp(result)
        log_recover.dropna(inplace=True)
        #print('log_recover结果序列值为%s'%log_recover)
        '''
        #数据展示
        ts = data[4][log_recover.index]  # 过滤没有预测的记录
        plt.figure(facecolor='white')
        log_recover.plot(color='blue', label='Predict')
        ts.plot(color='red', label='Original')
        plt.legend(loc='best')
        plt.title('RMSE: %.4f'% np.sqrt(sum((log_recover-ts)**2)/ts.size))
        plt.show()
        '''

        log_recover1, log_recover2, log_recover3 = predict_new(adf_tre, adf_season, 5, model_tre, data_train, log_recover_tre, adf_tre_num)
        print('预测值：\n', log_recover1)
        print('置信区间下限：\n', log_recover2)
        print('置信区间上限：\n', log_recover3)

        #数据展示'{:g}'.format(num)

        ts = data[log_recover1.index]  # 过滤没有预测原始数据的记录
        print(log_recover1.index)
        print(ts)
        print(log_recover1)
        #input('>>>>>>>>>>finally')
        plt.figure(facecolor='white')
        log_recover1.plot(color='blue', label='Predict')
        #log_recover2[-100:].plot(color='green', label='pre_low')
        #log_recover3[-100:].plot(color='green', label='Pre_up')
        ts.plot(color='red', label='Original')
        plt.legend(loc='best')
        plt.title('RMSE: %.4f'% np.sqrt(sum((log_recover1[-5:]-ts[-5:])**2)/ts[-5:].size))
        plt.show()
    else:
        print('模型残差不是白噪声，需进一步处理数据')







