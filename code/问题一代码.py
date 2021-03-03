# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 09:53:03 2021

@author: WCL
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
import os
from tqdm import tqdm
from sklearn import metrics
import torch
from torch import nn
import random
from fbprophet import Prophet
from statsmodels.tsa.arima_model import ARMA
import lightgbm as lgb
from sklearn.model_selection import KFold
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

df1= pd.read_csv(r"../input/6345234/shangxing.csv")
df2= pd.read_csv(r"../input/6345234/xiaxing.csv")
code = pd.read_csv(r"../input/6345234/code.csv")

df_1 = np.argmin(np.array(df1),axis=1)
df_2 = np.argmin(np.array(df2),axis=1)

# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def data_process(df):
    df.loc[df['日期']=='018-04-04',"日期"] = '2018/4/4'
    df.loc[df['日期']=='018-04-01',"日期"] = '2018/4/1'
    df.loc[df['日期']=='018-04-09',"日期"] = '2018/4/9'
    df.loc[df['日期']=='018-04-02',"日期"] = '2018/4/2'
    df.loc[df['日期']=='018-04-10',"日期"] = '2018/4/10'
    df.loc[df['日期']=='018-04-08',"日期"] = '2018/4/8'
    df.loc[df['日期']=='018-04-03',"日期"] = '2018/4/3'
    df['date'] = (df['日期']+' '+df['时间']).apply(lambda x:pd.to_datetime(x))
    df.sort_values("date",inplace=True)
    df = df.drop_duplicates().reset_index(drop=True)

    return df

def data_splits(df,n_time):
    train = df.iloc[:-n_time]
    valid = df.iloc[-n_time:]
    return train,valid

def metrics_(y_true, y_pred, metric):
    if metric=='MSE':
        MSE = metrics.mean_squared_error(y_true, y_pred)
        return MSE
    elif metric=='RMSE':
        RMSE = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
        return RMSE
    elif metric=='MAE':
        MAE = metrics.mean_absolute_error(y_true, y_pred)
        return MAE
    else:
        r2 = metrics.r2_score(y_true, y_pred)
        return r2
    
def time_process(data_df):
    date_ = data_df['date'].copy()
    data_df["hour"] = date_.dt.hour
    data_df["day"] = date_.dt.day
    data_df['dayofweek'] = date_.dt.dayofweek
    data_df['weekofyear'] = date_.dt.week
    return data_df

# 构造时间特征
def get_time_fe(data, col, n, one_hot=False, drop=True):
    data[col + '_sin'] = round(np.sin(2*np.pi / n * data[col]), 6)
    data[col + '_cos'] = round(np.cos(2*np.pi / n * data[col]), 6)
    if one_hot:
        ohe = OneHotEncoder()
        X = OneHotEncoder().fit_transform(data[col].values.reshape(-1, 1)).toarray()
        df = pd.DataFrame(X, columns=[col + '_' + str(int(i)) for i in range(X.shape[1])])
        data = pd.concat([data, df], axis=1)
    if drop:
        data = data.drop(col, axis=1)
    return data

# 构造过去 n 天的统计数据
def get_statis_n_days_num(data, col, col1,n):
    data['avg_'+ str(n) +'_days_' + col1] = data[col].rolling(window=n).mean()
    data['median_'+ str(n) +'_days_' + col1] = data[col].rolling(window=n).median()
    data['max_'+ str(n) +'_days_' + col1] = data[col].rolling(window=n).max()
    data['min_'+ str(n) +'_days_' + col1] = data[col].rolling(window=n).min()
    data['std_'+ str(n) +'_days_' + col1] = data[col].rolling(window=n).std()
    data['skew_'+ str(n) +'_days_' + col1] = data[col].rolling(window=n).skew()
    data['kurt_'+ str(n) +'_days_' + col1] = data[col].rolling(window=n).kurt()
    data['q1_'+ str(n) +'_days_' + col1] = data[col].rolling(window=n).quantile(0.25)
    data['q3_'+ str(n) +'_days_' + col1] = data[col].rolling(window=n).quantile(0.75)
    return data

def shfit_fe(tmp,tar,col1):
    for i in range(1,24):
        tmp[f'{col1}_shifts_{i}'] = tmp[tar].shift(i)
    return tmp

def get_target(tmp,tar,col1):
    for i in range(1,169):
        tmp[f'{col1}_{i}'] = tmp[tar].shift(-i)
    return tmp

def processing(df,code,name,cut,is_test):
    if is_test:
        df,valid = data_splits(df,7*24)
        #对数据进行去重操作
        df = df.drop_duplicates().reset_index(drop=True)
        df = df.fillna(method='ffill')
        #统计数据的日期数量存进字典
        t = dict(df['日期'].value_counts())
        data = pd.DataFrame({'日期': list(t.keys()),'num': list(t.values())})
        #删除一天中小于12条数据的天
        data = data.loc[data['num']!=24]
        data0 = data.loc[data['num']<12]
        columns = data0['日期'].values
        df = df[~df.日期.isin(columns)]
        #提取出数据中一天中12-24条数据的天
        data1 = data.loc[data['num']>=12]
        columns = data1['日期'].values
        for i in range(len(columns)):
            x = pd.date_range(start=columns[i],periods=24,freq='h')
            x = pd.DataFrame(x,columns=['date'])
            df = pd.merge(df,x,how='outer')
            df['日期'] = df["date"].apply(lambda x : x.strftime('%Y-%m-%d'))
            df['时间'] = df["date"].apply(lambda x : x.strftime('%H:%M:%S'))
            df['小区编号'] = code
            df = df.sort_values(by=['时间','日期'])  # 先按时间这一列进行排序
            df = df.reset_index(drop=True)
            #向前填充
            df = df.fillna(method='ffill')
            #再给他变回来
            df = df.sort_values(by=['date'])  # 先按时间这一列进行排序
            df = df.reset_index(drop=True)
        #对其异常值做一个处理
        data_x = df[name].dropna()
        data_cut = np.percentile(data_x,cut)
        df[name][df[name] >=  data_cut] =  data_cut
        df = pd.concat([df,valid],axis=0).reset_index(drop=True)
        return df
    else:
        #对数据进行去重操作
        df = df.drop_duplicates().reset_index(drop=True)
        df = df.fillna(method='ffill')
        #统计数据的日期数量存进字典
        t = dict(df['日期'].value_counts())
        data = pd.DataFrame({'日期': list(t.keys()),'num': list(t.values())})
        #删除一天中小于12条数据的天
        data = data.loc[data['num']!=24]
        data0 = data.loc[data['num']<12]
        columns = data0['日期'].values
        df = df[~df.日期.isin(columns)]
        #提取出数据中一天中12-24条数据的天
        data1 = data.loc[data['num']>=12]
        columns = data1['日期'].values
        for i in range(len(columns)):
            x = pd.date_range(start=columns[i],periods=24,freq='h')
            x = pd.DataFrame(x,columns=['date'])
            df = pd.merge(df,x,how='outer')
            df['日期'] = df["date"].apply(lambda x : x.strftime('%Y-%m-%d'))
            df['时间'] = df["date"].apply(lambda x : x.strftime('%H:%M:%S'))
            df['小区编号'] = code
            df = df.sort_values(by=['时间','日期'])  # 先按时间这一列进行排序
            df = df.reset_index(drop=True)
            #向前填充
            df = df.fillna(method='ffill')
            #再给他变回来
            df = df.sort_values(by=['date'])  # 先按时间这一列进行排序
            df = df.reset_index(drop=True)
        #对其异常值做一个处理
        data_x = df[name].dropna()
        data_cut = np.percentile(data_x,cut)
        df[name][df[name] >=  data_cut] =  data_cut
        return df

def Prophet_run(df=None ,fore_range=None, target=None, is_test=None):
    df = df.dropna().reset_index(drop=True)
    if is_test:
        #划分数据
        train,valid = data_splits(df,7*24)
        df_ = train[['date',target]].copy()
        df_.columns = ['ds', 'y']
        model = Prophet()
        model.fit(df_)
        #预测验证数据
        future_val = pd.DataFrame(list(valid['date']))
        future_val.columns = ['ds']
        forecast = model.predict(future_val)
        RMSE = metrics_(valid[target], forecast['yhat'], 'RMSE')
#         #预测
#         df_fog = df[['date',target]].copy()
#         df_fog.columns = ['ds', 'y']
#         model_fog = Prophet()
#         model_fog.fit(df_fog)
#         future = pd.DataFrame(list(fore_range))
#         future.columns = ['ds']
#         forecast_fog = model_fog.predict(future)
        return RMSE

    else:
        df_ = df[['date',target]].copy()
        df_.columns = ['ds', 'y']
        model = Prophet()
        model.fit(df_)
        #预测
        df_fog = df[['date',target]].copy()
        df_fog.columns = ['ds', 'y']
        model_fog = Prophet()
        model_fog.fit(df_fog)
        future = pd.DataFrame(list(fore_range))
        future.columns = ['ds']
        forecast_fog = model_fog.predict(future)
        return forecast_fog['yhat']

class forecastNet():
    """
    Class for the densely connected hidden cells version of the model
    """
    def __init__(self, seed_value=None, path=None, str_=None, IS_test=None, WindowLength=None, 
                 hidden_dim=None, epochs=None, batch_size=None,code=None):
        """
        Constructor
        :param seed_value: tf的种子值
        :param path: 文件夹路径
        :param str_: 预测的目标字符串
        :param IS_test: 是否验证(True表示只验证，False表示只预测)
        :param WindowLength: 训练的时间步长
        :param hidden_dim: dense的神经元个数
        :param epochs: 训练的迭代次数
        :param batch_size: 训练的批次大小
        """
        self.is_test = IS_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.code = codes
        
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
        df = pd.read_csv(path+'/'+f'df_{code}.csv',index_col=0)
        df = self.data_process(df = df)
        df = self.processing(df=df,code=code,name=str_,cut=95,is_test=IS_test)
        trainX, trainY, data, sc = self.get_data(df = df, slidingWindowLength = WindowLength, 
                                                                  str_ = str_, is_test = IS_test)
        self.predicted = self.forecastNetDenseDenseGraph(Xtrain = trainX, Ytrain = trainY, train_data = data, 
                                                sc = sc, hiddenDim = hidden_dim, is_test = IS_test)
        
    def data_process(self, df=None):
        df.loc[df['日期']=='018-04-04',"日期"] = '2018/4/4'
        df.loc[df['日期']=='018-04-01',"日期"] = '2018/4/1'
        df.loc[df['日期']=='018-04-09',"日期"] = '2018/4/9'
        df.loc[df['日期']=='018-04-02',"日期"] = '2018/4/2'
        df.loc[df['日期']=='018-04-10',"日期"] = '2018/4/10'
        df.loc[df['日期']=='018-04-08',"日期"] = '2018/4/8'
        df.loc[df['日期']=='018-04-03',"日期"] = '2018/4/3'
        df['date'] = (df['日期']+' '+df['时间']).apply(lambda x:pd.to_datetime(x))
        df.sort_values("date",inplace=True)
#         df = df.fillna(method='ffill')
#         df = df.drop_duplicates().reset_index(drop=True)
        
        return df
    
    def processing(self, df=None,code=None,name=None,cut=None,is_test=None):
        if is_test:
            df,valid = self.data_splits(df=df,n_time=7*24)
            #对数据进行去重操作
            df = df.drop_duplicates().reset_index(drop=True)
            df = df.fillna(method='ffill')
            #统计数据的日期数量存进字典
            t = dict(df['日期'].value_counts())
            data = pd.DataFrame({'日期': list(t.keys()),'num': list(t.values())})
            #删除一天中小于12条数据的天
            data = data.loc[data['num']!=24]
            data0 = data.loc[data['num']<12]
            columns = data0['日期'].values
            df = df[~df.日期.isin(columns)]
            #提取出数据中一天中12-24条数据的天
            data1 = data.loc[data['num']>=12]
            columns = data1['日期'].values
            for i in range(len(columns)):
                x = pd.date_range(start=columns[i],periods=24,freq='h')
                x = pd.DataFrame(x,columns=['date'])
                df = pd.merge(df,x,how='outer')
                df['日期'] = df["date"].apply(lambda x : x.strftime('%Y-%m-%d'))
                df['时间'] = df["date"].apply(lambda x : x.strftime('%H:%M:%S'))
                df['小区编号'] = code
                df = df.sort_values(by=['时间','日期'])  # 先按时间这一列进行排序
                df = df.reset_index(drop=True)
                #向前填充
                df = df.fillna(method='ffill')
                #再给他变回来
                df = df.sort_values(by=['date'])  # 先按时间这一列进行排序
                df = df.reset_index(drop=True)
            #对其异常值做一个处理
            data_x = df[name].dropna()
            data_cut = np.percentile(data_x,cut)
            df[name][df[name] >=  data_cut] =  data_cut
            df = pd.concat([df,valid],axis=0).reset_index(drop=True)
            return df
        else:
            #对数据进行去重操作
            df = df.drop_duplicates().reset_index(drop=True)
            df = df.fillna(method='ffill')
            #统计数据的日期数量存进字典
            t = dict(df['日期'].value_counts())
            data = pd.DataFrame({'日期': list(t.keys()),'num': list(t.values())})
            #删除一天中小于12条数据的天
            data = data.loc[data['num']!=24]
            data0 = data.loc[data['num']<12]
            columns = data0['日期'].values
            df = df[~df.日期.isin(columns)]
            #提取出数据中一天中12-24条数据的天
            data1 = data.loc[data['num']>=12]
            columns = data1['日期'].values
            for i in range(len(columns)):
                x = pd.date_range(start=columns[i],periods=24,freq='h')
                x = pd.DataFrame(x,columns=['date'])
                df = pd.merge(df,x,how='outer')
                df['日期'] = df["date"].apply(lambda x : x.strftime('%Y-%m-%d'))
                df['时间'] = df["date"].apply(lambda x : x.strftime('%H:%M:%S'))
                df['小区编号'] = code
                df = df.sort_values(by=['时间','日期'])  # 先按时间这一列进行排序
                df = df.reset_index(drop=True)
                #向前填充
                df = df.fillna(method='ffill')
                #再给他变回来
                df = df.sort_values(by=['date'])  # 先按时间这一列进行排序
                df = df.reset_index(drop=True)
            #对其异常值做一个处理
            data_x = df[name].dropna()
            data_cut = np.percentile(data_x,cut)
            df[name][df[name] >=  data_cut] =  data_cut
            return df
        
    def data_splits(self, df=None,n_time=None):
        train = df.iloc[:-n_time]
        valid = df.iloc[-n_time:]
        return train,valid

    def metrics_(self, y_true=None, y_pred=None, metric=None):
        if metric=='MSE':
            MSE = metrics.mean_squared_error(y_true, y_pred)
            return MSE
        elif metric=='RMSE':
            RMSE = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
            return RMSE
        elif metric=='MAE':
            MAE = metrics.mean_absolute_error(y_true, y_pred)
            return MAE
        else:
            r2 = metrics.r2_score(y_true, y_pred)
            return r2
        
    def get_data(self, df=None, slidingWindowLength=None, str_=None, is_test=None):
        """
        该函数用于得到forecastNet训练和测试数据
        :param df: 输入数据
        :param slidingWindowLength: 训练数据时间步
        :param str_: 输入需要预测单变量字符串
        :param is_test: 是否测试误差
        """
        if is_test:
            data = df[str_].values
            train_data, test_data = data[:-7*24],data[-7*24:]  #划分最后一周数据测试
            train_data = train_data.reshape(-1,1)
            #标准化
            sc = MinMaxScaler(feature_range=(0,1))
            train = sc.fit_transform(train_data).reshape(-1)
            #构建训练数据和标签
            trainX, trainY = list(), list()
            for i in range(len(train)-slidingWindowLength):
                x = train[i : i+slidingWindowLength]
                y = train[i+slidingWindowLength : i+slidingWindowLength+1]
                trainX.append(x.tolist())
                trainY.append(y.tolist())
            trainX, trainY = np.asarray(trainX), np.asarray(trainY)
            return trainX, trainY, train_data, test_data, sc
        else:
            data = df[str_].values
            data = data.reshape(-1,1)
            #标准化
            sc = MinMaxScaler(feature_range=(0,1))
            train = sc.fit_transform(data).reshape(-1)
            #构建训练数据和标签
            trainX, trainY = list(), list()
            for i in range(len(train)-slidingWindowLength):
                x = train[i : i+slidingWindowLength]
                y = train[i+slidingWindowLength : i+slidingWindowLength+1]
                trainX.append(x.tolist())
                trainY.append(y.tolist())
            trainX, trainY = np.asarray(trainX), np.asarray(trainY)
            return trainX, trainY, data, sc
        
    def forecastNetDenseDenseGraph(self, Xtrain=None, Ytrain=None, train_data=None, test_data=None, sc=None, 
                                   hiddenDim=None, is_test=None):
        '''
        该函数用于单变量全连接时变网络模型的构建
        :param X: 多步长单变量特征输入
        :param Y: 目标变量值
        :param hiddenDim: 每一个隐含层节点数
        :param outSeqLength: 输出多步长目标变量的长度
        :return: 时变深度前馈预测模型
        '''
        # 获取输入参数
        _, featureDims = Xtrain.shape
        # 创建输入
        inputs = tf.keras.Input(shape=(featureDims, ))
        # 第一个隐藏层
        hidden = tf.keras.layers.Dense(units=hiddenDim,
                                       activation="relu")(inputs)
        # 第二个隐藏层
        hidden = tf.keras.layers.Dense(units=hiddenDim,
                                       activation="relu")(hidden)
        # 第一交错输出
        outputs = tf.keras.layers.Dense(units=1, activation=None)(hidden)
        # 合并以上输出
        concat = tf.keras.layers.Concatenate(axis=-1)([inputs, hidden, outputs[:, 0:1]])
        # 注意力机制
        concatWeights = tf.keras.layers.Dense(units=concat.get_shape().as_list()[-1],
                                              activation="sigmoid",
                                              name="attention1")(concat)
        concatWighted = tf.keras.layers.Multiply()([concat, concatWeights])
        # 接下来第一个隐藏层
        hidden = tf.keras.layers.Dense(units=hiddenDim, activation="relu")(concatWighted)
        # 接下来第二个隐藏层
        hidden = tf.keras.layers.Dense(units=hiddenDim, activation="relu")(hidden)
        # 最后一层为全连接
        output = tf.keras.layers.Dense(units=1, activation=None)(hidden)
        # 合并输出
        outputs = tf.keras.layers.Concatenate(axis=-1)([outputs, output])

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    #     model.summary()
        history = model.fit(Xtrain, Ytrain, batch_size=self.batch_size, verbose=0, epochs=self.epochs, shuffle=False, use_multiprocessing=False)
        if is_test:
            X_test = np.array(train_data[-24:])
            for i in range(168):
                test = X_test[i:i+24]
                X_ = np.reshape(test, (test.shape[1],test.shape[0],1))
                predicts = model.predict(X_,batch_size=32)
                X_test = np.vstack((X_test, predicts[:,0]))
            predicted = sc.inverse_transform(X_test[-168:])
    #         print(test_data,predicted)
            absoluteError = self.metrics_(y_true = test_data, y_pred = predicted, metric = 'RMSE')
            return absoluteError
        else:
            X_test = np.array(train_data[-24:])
            for i in range(168):
                test = X_test[i:i+24]
                X_ = np.reshape(test, (test.shape[1],test.shape[0],1))
                predicts = model.predict(X_,batch_size=32)
                X_test = np.vstack((X_test, predicts[:,0]))
            predicted = sc.inverse_transform(X_test[-168:])
            return predicted

#_lstm
def get_lstm_data(df=None, slidingWindowLength=None, str_=None, is_test=None, device=None):
    """
    该函数用于得到lstm训练和测试数据
    :param df: 输入数据
    :param slidingWindowLength: 训练数据时间步
    :param str_: 输入需要预测单变量字符串
    :param is_test: 是否测试误差
    """
    if is_test:
        data = df[str_].values
        train_data, test_data = data[:-7*24],data[-7*24:]  #划分最后一周数据测试
        train_data = train_data.reshape(-1,1)
        #标准化
        sc = MinMaxScaler(feature_range=(0,1))
        train = sc.fit_transform(train_data).reshape(-1)
        train = torch.tensor(train, dtype=torch.float32, device=device).view(-1)
        #构建训练数据和标签
        inout_seq = []
        for i in range(len(train)-slidingWindowLength):
            x = train[i : i+slidingWindowLength]
            y = train[i+slidingWindowLength : i+slidingWindowLength+1]
            inout_seq.append((x ,y))
        return inout_seq, train, test_data, sc
    else:
        data = df[str_].values
        data = data.reshape(-1,1)
        #标准化
        sc = MinMaxScaler(feature_range=(0,1))
        train = sc.fit_transform(data).reshape(-1)
        train = torch.tensor(train, dtype=torch.float32, device=device).view(-1)
        #构建训练数据和标签
        inout_seq = []
        for i in range(len(train)-slidingWindowLength):
            x = train[i : i+slidingWindowLength]
            y = train[i+slidingWindowLength : i+slidingWindowLength+1]
            inout_seq.append((x ,y))
        return inout_seq, train, sc
    
def train_lstm_model(model=None, optimizer=None, loss_function=None, train_inout_seq=None, epochs=None, device=None):
    for i in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size, device=device),
                            torch.zeros(1, 1, model.hidden_layer_size, device=device))
            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()
#         if i%5 == 0:
#             print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

#     print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
    return model

def evaluate_lstm_test_set(model=None, train_data_normalized=None, train_window=None, pred_size=None, device=None, sc=None, is_test=None):
    if is_test:
        test_inputs = train_data_normalized[-train_window:].tolist()
        model.eval()
        for i in range(pred_size):
            seq = torch.tensor(test_inputs[-train_window:], dtype=torch.float32, device=device)
            with torch.no_grad():
                model.hidden = (torch.zeros(1, 1, model.hidden_layer_size, device=device),
                                torch.zeros(1, 1, model.hidden_layer_size, device=device))
                test_inputs.append(model(seq).item())
        actual_predictions = sc.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))
        RMSE1 = metrics_(test_data,actual_predictions, 'RMSE')
        return RMSE1
    else:
        test_inputs = train_data_normalized[-train_window:].tolist()
        model.eval()
        for i in range(pred_size):
            seq = torch.tensor(test_inputs[-train_window:], dtype=torch.float32, device=device)
            with torch.no_grad():
                model.hidden = (torch.zeros(1, 1, model.hidden_layer_size, device=device),
                                torch.zeros(1, 1, model.hidden_layer_size, device=device))
                test_inputs.append(model(seq).item())
        actual_predictions = sc.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))
        return actual_predictions
    
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=1, num_layers=None):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=self.num_layers)
        self.hidden_cell = (torch.zeros(self.num_layers,1,self.hidden_layer_size),torch.zeros(self.num_layers,1,self.hidden_layer_size))
        self.reg = nn.Sequential(
            nn.Linear(hidden_layer_size, 32),
#             nn.Dropout(p=0.5),
            nn.Tanh(),
            nn.Linear(32, output_size),
        )
    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.reg(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

def run_lgb(df_train, df_test,target,cols):
    model = lgb.LGBMRegressor(
                           objective = 'regression', #mse,rmse
                           boosting_type='gbdt',
                           num_leaves=32,
                           max_depth=8,
                           learning_rate=0.05,
                           n_estimators=200,
                           subsample=0.8,
                            min_data_in_leaf=350,
#                             max_bin=128,
                           feature_fraction=0.8,
                           random_state=2020,
                           n_jobs=-1,
                           metric='mse')

    preds = np.zeros(df_test.shape[0])
    kfold = KFold(n_splits=5,shuffle=True, random_state=2020)
    for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(df_train,target)):
        X_train = df_train.iloc[trn_idx]
        Y_train = target.iloc[trn_idx][cols]
        X_val = df_train.iloc[val_idx]
        Y_val = target.iloc[val_idx][cols]
        lgb_model = model.fit(X_train, 
                              Y_train,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train, Y_train), (X_val, Y_val)],
                              verbose=0,
                              eval_metric='mse',
                              early_stopping_rounds=50)
        preds += lgb_model.predict(df_test, num_iteration=lgb_model.best_iteration_)/5
    return preds

def lgb_run(df=None, str_=None, title_=None, is_test=None):

    df = get_time_fe(df, 'day', n=31, one_hot=False, drop=True)
    df = get_time_fe(df, 'dayofweek', n=7, one_hot=True, drop=True)
    df = get_time_fe(df, 'hour', n=12, one_hot=True, drop=True)
    df.sort_values("date",inplace=True)
    df = get_statis_n_days_num(df, str_, title_, n=24*1)
    df = get_statis_n_days_num(df, str_, title_, n=24*2)
    df = get_statis_n_days_num(df, str_, title_, n=24*3)
    df = get_statis_n_days_num(df, str_, title_, n=24*4)
    df = get_statis_n_days_num(df, str_, title_, n=24*5)
    df = get_statis_n_days_num(df, str_, title_, n=24*6)
    # df = shfit_fe(df,'上行业务量GB','shangxing')
    df = df.dropna().reset_index(drop=True)
    if is_test:
        #划分数据
        train_data,test_data = data_splits(df,7*24)
        train_data,test_data1 = train_data.iloc[:-1],train_data.iloc[-1]
        train_data = get_target(train_data, str_, title_)
        train_data = train_data.dropna().reset_index(drop=True)
        col = ['日期', '时间', '小区编号', '上行业务量GB', '下行业务量GB','date']
        target_cols = [f'{title_}_'+str(i) for i in range(1,169)]
        feats_cols = [i for i in df.columns if i not in target_cols+col]
        test_df = pd.DataFrame()
#         test_x = test_data.iloc[0]
        test_x = pd.DataFrame().append(test_data1, ignore_index=True)
        
        for i in target_cols:
            tests = run_lgb(train_data[feats_cols], test_x[feats_cols], train_data[target_cols], i)
            test_df[i] = tests
        RMSE = metrics_(test_data[str_], test_df.values.flatten(), 'RMSE')
        return RMSE,test_df
    else:
        train_data,test_data = df.iloc[:-1],df.iloc[-1]
        train_data = get_target(train_data, str_, title_)
        train_data = train_data.dropna().reset_index(drop=True)
        col = ['日期', '时间', '小区编号', '上行业务量GB', '下行业务量GB','date']
        target_cols = [f'{title_}_'+str(i) for i in range(1,169)]
        feats_cols = [i for i in df.columns if i not in target_cols+col]
        test_df = pd.DataFrame()
        test_x = pd.DataFrame().append(test_data, ignore_index=True)
        for i in target_cols:
            tests = run_lgb(train_data[feats_cols], test_x[feats_cols], train_data[target_cols], i)
            test_df[i] = tests
        return test_df

def Arma_run(df=None ,fore_range=None, target=None, is_test=None):
    df = df.dropna().reset_index(drop=True)
    if is_test:
        #划分数据
        train,valid = data_splits(df,7*24)
        armamodel = ARMA(train[target], order=(1,0))
        fit_result = armamodel.fit(disp = -1, method="css")
        predictions=fit_result.predict(start=len(train), end=len(train)+len(valid)-1, dynamic=False)
        RMSE1 = metrics_(valid[target], predictions, 'RMSE')
        #预测
        df_fog = df[['date',target]].copy()
        df_fog.columns = ['ds', 'y']
        model = ARMA(df_fog['y'], order=(1,0))
        model_fog = model.fit(disp = -1, method="css")
        future = pd.DataFrame(list(fore_range))
        future.columns = ['ds']
        forecast_fog = model_fog.predict(start=len(df), end=len(df)+len(future)-1, dynamic=False)
        return RMSE1,forecast_fog

    else:
        df_fog = df[['date',target]].copy()
        df_fog.columns = ['ds', 'y']
        model = ARMA(df_fog['y'], order=(1,0))
        model_fog = model.fit(disp = -1, method="css")
        future = pd.DataFrame(list(fore_range))
        future.columns = ['ds']
        forecast_fog = model_fog.predict(start=len(df), end=len(df)+len(future)-1, dynamic=False)
        return forecast_fog
    
def prophet(path = None,str_ = None,code = None,fore_range=None):
    IS_test = False
    df = pd.read_csv(path+'/'+f'df_{code}.csv',index_col=0)
    df = data_process(df)
    fore_range = sub.loc[sub['小区编号']==code,"date"]
    df = processing(df,code,str_,95,is_test=IS_test)
    predictions = Prophet_run(df,fore_range,target=str_, is_test=IS_test)
    return predictions,fore_range    
    
def lstm(path = None,str_ = None,code = None):
    pred_size = 168
    train_window = 24
    num_layers = 1
    IS_test = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(2020)
    df = pd.read_csv(path+'/'+f'df_{code}.csv',index_col=0)
    model = LSTM(num_layers=num_layers).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    df = data_process(df = df)
    df = processing(df,code,str_,95,IS_test)
    inout_seq, train, sc = get_lstm_data(df = df, slidingWindowLength = train_window, 
                                                str_ = str_, is_test = IS_test, device=device)
    model = train_lstm_model(model=model, optimizer=optimizer, loss_function=loss_function, 
                        train_inout_seq=inout_seq, epochs=20, device=device)
    predictions = evaluate_lstm_test_set(model=model, train_data_normalized = train, train_window=train_window, 
                      pred_size=pred_size, device=device, is_test = IS_test, sc=sc)
    return predictions    
    
def lightgbm(path = None,str_ = None,code = None,title_ = None):
    IS_test = False
    df = pd.read_csv(path+'/'+f'df_{code}.csv',index_col=0)
    df = data_process(df)
    df = processing(df,code,str_,95,IS_test)
    df = time_process(df)
    test_df = lgb_run(df=df, str_=str_, title_=title_, is_test=IS_test)
    predictions = test_df.values.flatten()  

    return predictions    
    
def arma(path = None,str_ = None,code = None,fore_range=None):
    IS_test = False
    df = pd.read_csv(path+'/'+f'df_{code}.csv',index_col=0)
    df = data_process(df)
    fore_range = sub.loc[sub['小区编号']==code,"date"]
    df = processing(df,code,str_,95,IS_test)
    predictions = Arma_run(df ,fore_range, target=str_, is_test=IS_test)
    return predictions,fore_range    
    
path = '../input/643523432/1/新建文件夹1'
sub = pd.read_csv('../input/434132/2.csv',encoding='gbk')
sub['date'] = (sub['日期']+' '+sub['时间']).apply(lambda x:pd.to_datetime(x))

str_ = '上行业务量GB'
date_range = pd.date_range(start='20180420',end='20180427', freq='H', closed='left')
predicts_shang = pd.DataFrame(columns = ['code',str_])
for i,j in tqdm(enumerate(df_1)):
    codes = code['code'].iloc[i]
    if codes in [9579,8954]:
        j = 0
    if j == 0:
        predictions,fore_range = prophet(path = path,str_ = str_,code = codes)
        _ = pd.DataFrame({'code':codes,'date':list(fore_range),str_:predictions}).reset_index(drop=True)
        predicts_shang = predicts_shang.append(_)

    elif j == 1:
        predictions = lstm(path = path,str_ = str_,code = codes)
        _ = pd.DataFrame({'code':codes,'date':date_range,str_:predictions.flatten()}).reset_index(drop=True)
        predicts_shang = predicts_shang.append(_)

    elif j == 2:
        predictions = lightgbm(path = path,str_ = str_,code = codes,title_ = 'shangxing')
        _ = pd.DataFrame({'code':codes,'date':date_range,str_:predictions}).reset_index(drop=True)
        predicts_shang = predicts_shang.append(_)
 
    elif j == 3:
        predictions = forecastNet(seed_value=20, path=path, str_=str_, 
                                  IS_test=False, WindowLength=24, hidden_dim=45, epochs=32, batch_size=32,code = codes)
        _ = pd.DataFrame({'code':codes,'date':date_range,str_:predictions.predicted.flatten()}).reset_index(drop=True)
        predicts_shang = predicts_shang.append(_)

    else:
        predictions,fore_range = arma(path = path,str_ = str_,code = codes)
        _ = pd.DataFrame({'code':codes,'date':list(fore_range),str_:predictions}).reset_index(drop=True)
        predicts_shang = predicts_shang.append(_)

strs_ = '下行业务量GB'
predicts_xia = pd.DataFrame(columns = ['code',strs_])
for i,j in tqdm(enumerate(df_2)):
    codes = code['code'].iloc[i]
    if j == 0:
        predictions,fore_range = prophet(path = path,str_ = strs_,code = codes)
        _ = pd.DataFrame({'code':codes,'date':list(fore_range),strs_:predictions}).reset_index(drop=True)
        predicts_xia = predicts_xia.append(_)
    elif j == 1:
        predictions,fore_range = arma(path = path,str_ = strs_,code = codes)
        _ = pd.DataFrame({'code':codes,'date':list(fore_range),strs_:predictions}).reset_index(drop=True)
        predicts_xia = predicts_xia.append(_)
    elif j == 2:
        predictions = forecastNet(seed_value=20, path=path, str_=strs_, 
                                  IS_test=False, WindowLength=24, hidden_dim=45, epochs=32, batch_size=32,code = codes)
        _ = pd.DataFrame({'code':codes,'date':date_range,strs_:predictions.predicted.flatten()}).reset_index(drop=True)
        predicts_xia = predicts_xia.append(_)
    elif j == 3:
        predictions = lightgbm(path = path,str_ = strs_,code = codes,title_ = 'xiaxing')
        _ = pd.DataFrame({'code':codes,'date':date_range,strs_:predictions}).reset_index(drop=True)
        predicts_xia = predicts_xia.append(_)
    else:
        predictions = lstm(path = path,str_ = strs_,code = codes)
        _ = pd.DataFrame({'code':codes,'date':date_range,strs_:predictions.flatten()}).reset_index(drop=True)
        predicts_xia = predicts_xia.append(_)

#数据合并
predicts_shang['id'] = predicts_shang['code'].astype(str)+predicts_shang['date'].astype(str)
predicts_xia['id'] = predicts_xia['code'].astype(str)+predicts_xia['date'].astype(str)
sub['id'] = sub['小区编号'].astype(str)+sub['date'].astype(str)
sub = sub.drop(['上行业务量GB', '下行业务量GB'],axis=1)
predicts_xia = predicts_xia.rename(columns={'上行业务量GB':'下行业务量GB'})
sub = pd.merge(sub, predicts_shang[['id','上行业务量GB']].reset_index(drop=True),on=['id'],how='left')
sub = pd.merge(sub, predicts_xia[['id','下行业务量GB']].reset_index(drop=True),on=['id'],how='left')
del sub['date']
del sub['id']
sub.loc[sub['上行业务量GB']<0,"上行业务量GB"] = 0
sub.loc[sub['下行业务量GB']<0,"下行业务量GB"] = 0
sub.to_csv('附件2：短期验证选择的小区数据集.csv',index=False)


