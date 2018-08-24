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

##模型一
class ModelDecomp(object):
    def __init__(self, ts, test_size):
        self.train = ts[:len(ts)-test_size]##训练集
        self.test = ts[-test_size:]##测试集

    #平稳性检验
    def adf_test(self, ts):
        adf = adfuller(ts, autolag='AIC')
        return adf

    #白噪声检验
    def acorr_test(self, ts):
        lg = acorr_ljungbox(ts, lags=1)
        return lg

    #最优差分
    def diff_test(self, ts):
        '''
        :param ts: 原始序列
        :return: adf值，差分次数，差分后平稳序列,经过差分的全部序列
        '''
        adf = self.adf_test(ts)
        diff_num = 0
        ts_diff = ts
        global ts_diff_all
        ts_diff_all = []
        while adf[1] >= 0.05:
            diff_num = diff_num+1
            ts_diff = ts.diff(diff_num).dropna()
            ts_diff_all.append(ts_diff)#添加到差分数据列表中
            adf = self.adf_test(ts_diff)
        print(ts_diff)
        print(u'原始序列经过%s阶差分后归于平稳，p值为%s' %(diff_num,adf[1]))
        return adf, diff_num, ts_diff, ts_diff_all

    #模型分解
    def data_prepro(self, ts):
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
    def model_test(self, ts, d, maxlag):
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
    def predict_diff_recover(self, predict_value, ts_diff_all, d):
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
    def predict_new(self, adf_tre, seasonal, n, model, data, log_recover_tre, d):
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

    ##样本展示
    def data_show(self,**akwad):
        pass


if __name__ =='__main__':
    datafile = 'data2.xls'
    data = pd.read_excel(datafile, index_col='date')##读取数据
    data.index = pd.to_datetime(data.index, format='%Y%m%d')##设置索引样式
    zb_code = data.columns##获取指标列表
    #md = ModelDecomp(data,test_size=5)
    for value in zb_code:
        ts = data[value]

    md = ModelDecomp(ts = data['ZJ027161'],test_size=5)#调用模型类

