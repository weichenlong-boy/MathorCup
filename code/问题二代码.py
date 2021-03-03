
def data_pro(df):
    df.loc[df['日期']=='018-04-04',"日期"] = '2018/4/4'
    df.loc[df['日期']=='018-04-01',"日期"] = '2018/4/1'
    df.loc[df['日期']=='018-04-09',"日期"] = '2018/4/9'
    df.loc[df['日期']=='018-04-02',"日期"] = '2018/4/2'
    df.loc[df['日期']=='018-04-10',"日期"] = '2018/4/10'
    df.loc[df['日期']=='018-04-08',"日期"] = '2018/4/8'
    df.loc[df['日期']=='018-04-03',"日期"] = '2018/4/3'
    df['date'] = (df['日期']+' '+df['时间']).apply(lambda x:to_datetime(x))
    df.sort_values("date",inplace=True)
    return df
    
def processing(df,code,name):
    #对数据进行去重操作
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.fillna(method='ffill')
    #统计数据的日期数量存进字典
    t = dict(df['日期'].value_counts())
    data = pd.DataFrame({'日期': list(t.keys()),'num': list(t.values())})
    data = data.loc[data['num']!=24]
    columns = data['日期'].values
    v = pd.DataFrame(columns)
    v.columns = ['columns']
    v['columns'] = pd.to_datetime(v['columns'])
    v['columns'] = v['columns'].apply(lambda x : x.strftime('%Y-%m-%d'))
    if len(df)!=0:
        for i in range(len(columns)):
            x = pd.date_range(start=v['columns'].iloc[i],periods=24,freq='h')
            x = pd.DataFrame(x,columns=['date'])
            df = pd.merge(df,x,how='outer')
            df['日期'] = df["date"].apply(lambda x : x.strftime('%Y-%m-%d'))
            df['时间'] = df["date"].apply(lambda x : x.strftime('%H:%M:%S'))
            df['小区编号'] = code
            #均值填充
            df_q = df[df['日期']==v['columns'].iloc[i]]
            df = df.fillna(df_q.mean())
            #df = df.fillna(df.mean())
            df = df.reset_index(drop=True)
        return df
    else:
        return df

def combine(df,x,y):
    combine = df[x].groupby(df[y]).agg('sum')
    combine = pd.DataFrame({
        'ds': combine.index,
        'y': combine.values,
  })
    return combine

def model_prophet(df,time1,time2):
    model = Prophet()
    model.fit(df)
    future = pd.date_range(time1,time2)
    future = pd.DataFrame(future)
    future.columns = ['ds']
    future['ds']= to_datetime(future['ds'])
    forecast = model.predict(future)
    return forecast['yhat']

path = '/content/drive/MyDrive/data/长期验证'
sub = pd.read_csv('/content/drive/MyDrive/data/长期验证.csv',encoding='gbk')
future = pd.date_range('2020-11-01','2020-11-25')
future = pd.DataFrame(future)
future.columns = ['ds']
future['ds']= to_datetime(future['ds'])
files = os.listdir(path)
Code = []
#创建一个空的datafrmae
data = pd.DataFrame(columns = ["code","日期", '上行业务量GB'])
#上行
for i in tqdm(files):
    code = int(os.path.splitext(i)[0][3:])
    df = pd.read_csv(path+'/'+i)
    if len(df)!=0:
        df = data_pro(df)
        df1 = processing(df,code,'上行业务量GB')
        df_up = combine(df1,'上行业务量GB','日期')
        #预测上行业务量GB
        if len(df_up)!=0 and len(df_up)!=1:
            forecast1 = model_prophet(df_up,'2020-11-01','2020-11-25')
            df_x = pd.DataFrame({
                  'code': code,
                  '日期':future['ds'],
                  '上行业务量GB': list(forecast1),
                  })
            data = data.append(df_x)
          #print(data.tail())
        else:
          #将需要最后单独处理的小区号提出
          Code.append(code)
    else:
        Code.append(code)
        
time = data['日期'].head(25)
df_none_up = pd.DataFrame(columns = ["code", "日期","上行业务量GB"])
for i in range(len(time)):
    data1 = data[data['日期']==time.iloc[i]]
    #将单独异常的小区号赋予一个新的dataframe，上行业务量GB初始为0
    df_x = pd.DataFrame({
    'code': Code,
    '日期': time.iloc[i],
    '上行业务量GB': 0,
    })
    data1 = data1.append(df_x)
    #取出上行业务量GB大于0的均值并赋予上行业务量GB小于等于0的值
    data2 = data1[data1['上行业务量GB']>0]
    q = data2['上行业务量GB'].mean()
    data1.loc[data1.上行业务量GB<=0,'上行业务量GB']=q
    df_none_up = df_none_up.append(data1)

future = pd.date_range('2020-11-01','2020-11-25')
future = pd.DataFrame(future)
future.columns = ['ds']
future['ds']= to_datetime(future['ds'])
files = os.listdir(path)
Code = []
#创建一个空的datafrmae
data = pd.DataFrame(columns = ["code","日期", '下行业务量GB'])
#下行
for i in tqdm(files):
    code = int(os.path.splitext(i)[0][3:])
    df = pd.read_csv(path+'/'+i)
    if len(df)!=0:
        df = data_pro(df)
        df2 = processing(df,code,'下行业务量GB')
        df_down = combine(df2,'下行业务量GB','日期')
        #预测下行业务量GB
        if len(df_down)!=0 and len(df_down)!=1:
            forecast2 = model_prophet(df_down,'2020-11-01','2020-11-25')
            df_x = pd.DataFrame({
                  'code': code,
                  '日期':future['ds'],
                  '下行业务量GB': list(forecast2),
                  })
            data = data.append(df_x)
          #print(data.tail())
        else:
          #将需要最后单独处理的小区号提出
          Code.append(code)
    else:
        Code.append(code)
        
time = data['日期'].head(25)
df_none_down = pd.DataFrame(columns = ["code", "日期","下行业务量GB"])
for i in range(len(time)):
    data1 = data[data['日期']==time.iloc[i]]
    #将单独异常的小区号赋予一个新的dataframe，上行业务量GB和下行业务量GB初始均为0
    df_x = pd.DataFrame({
    'code': Code,
    '日期': time.iloc[i],
    '下行业务量GB': 0
    })
    data1 = data1.append(df_x)
    #取出下行业务量GB大于0的均值并赋予下行业务量GB小于等于0的值
    data3 = data1[data1['下行业务量GB']>0]
    p = data3['上行业务量GB'].mean()
    data1.loc[data1.下行业务量GB<=0,'下行业务量GB']=p
    df_none_down = df_none_down.append(data1)

df_none = pd.merge(df_none_up,df_none_down,on=['code','日期'])
df_none.columns = ['小区编号', '日期', '上行业务量GB', '下行业务量GB']
#将日期格式修改为提交格式集
df_none['日期'] = pd.to_datetime(df_none['日期'])
df_none['日期'] = df_none['日期'].apply(lambda x : x.strftime('%Y/%-m/%-d'))

#待预测文件并不是每一个小区都为25天，取预测结果与提交结果的交集
sub=pd.read_csv('./长期验证.csv',encoding='gbk')
submit = sub.drop(['上行业务量GB', '下行业务量GB'],axis=1)
intersection_result = pd.merge(submit, df_none)

#将预测结果上行业务量GB大于下行业务量GB的数据进行后处理
import numpy as np
path = '/content/drive/MyDrive/data/长期验证'
t = intersection_result.loc[intersection_result.上行业务量GB>intersection_result.下行业务量GB]
for i in range(len(t)):
    code = t.iloc[i]['小区编号']
    df = pd.read_csv(path+'/df_'+str(code)+'.csv')
    bs = np.sum(df.下行业务量GB)/np.sum(df.上行业务量GB)
    t.iloc[i,3:4] = t.iloc[i,2:3].values*bs
    
x1 = intersection_result.loc[intersection_result.上行业务量GB<=intersection_result.下行业务量GB]
data = x1.append(t)
data = data.reset_index(drop=True)
submit = sub.drop(['上行业务量GB', '下行业务量GB'],axis=1)
result = pd.merge(submit, data)
result.to_csv('./submit.csv',index=0)