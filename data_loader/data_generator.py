import numpy as np
import pandas as pd
import datetime
import os
from sklearn.preprocessing import MinMaxScaler
def csv_read(csv_name='bundang_ratio_ver.csv',encoding=None):
    raw_data=pd.read_csv(csv_name,encoding=encoding,thousands=',')
    a=[]
    for i in raw_data['계약일']:
        i=str(i).zfill(2) #한 자릿수 숫자 앞에 0을 넣음
        a.append(i)
    raw_data['계약일']=a
    raw_data['계약년월']=raw_data['계약년월'].astype(str)
    raw_data['date']=raw_data['계약년월']+raw_data['계약일']
    raw_data['ppa']=raw_data['거래금액(만원)']/raw_data['전용면적(㎡)']
    raw_data=raw_data[['date','ppa','건축년도','apt_num','psycho_num','price_num','floor_ratio','scale_ratio']]
    #column 이름 재설정
    raw_data['built']=raw_data['건축년도']
    raw_data['built']=raw_data['date'].astype(int)/10000-raw_data['built']
    raw_data['built']=raw_data['built'].astype(int)
    raw_data=raw_data.drop(raw_data[['건축년도']],axis=1)
    raw_data=raw_data[['date','ppa','built','apt_num','psycho_num','price_num','floor_ratio','scale_ratio']]

    b=[]
    for j in raw_data['date']:
        j=datetime.datetime.strptime(j,'%Y%m%d').date()
        b.append(j)
    start=datetime.date(2010,1,1)
    c=[]
    for i in range(len(b)):
        c.append(int((b[i]-start).days))
    raw_data['date_num']=c
    raw_data=raw_data[raw_data.apt_num!=0]
    raw_data=raw_data[raw_data.price_num!=0]
    raw_data=raw_data[raw_data.psycho_num!=0]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data=raw_data[['date_num','built','apt_num','psycho_num','price_num','floor_ratio','scale_ratio']]
    labels=raw_data[['ppa']]
    return data,labels


class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        inn,y=csv_read(r'C:\Users\00001234\mlp10\data_loader\bundang_ratio_ver.csv',encoding='Ansi')
        
        self.input = np.array(inn.values)
        self.y = np.array(y.values)



    def next_batch(self, batch_size):
#        idx = np.random.choice(500, batch_size)
        idx = np.random.choice(len(self.input), batch_size,replace=False)
        yield self.input[idx], self.y[idx]

    def get_full_dataset(self):
        return self.input, self.y
