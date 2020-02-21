import pandas as pd
import numpy as np
def csv_read(csv_name='csv/2010_2019_bundang_v0.csv',encoding=None):
    raw_data=pd.read_csv(csv_name,encoding=encoding,thousands=',')
    a=[]
    for i in raw_data['계약일']:
        i=str(i).zfill(2) #한 자릿수 숫자 앞에 0을 넣음
        a.append(i)
    raw_data['계약일']=a
    raw_data['계약년월']=raw_data['계약년월'].astype(str)
    raw_data['date']=raw_data['계약년월']+raw_data['계약일']
    raw_data=raw_data.drop(['시군구','번지','본번','부번','단지명','계약년월','계약일','도로명'], axis=1)
    #column 이름 재설정
    raw_data.columns=['area','price','floor','built','apt_num','psycho_num','price_num','date']
    raw_data['ppa']=raw_data['price']/raw_data['area']
    raw_data['scale']=raw_data['area']/2.6
    c=[]
    #평형 분할
    for area in raw_data['scale']:
        if area<=20:
            c.append(1.)
        elif area>20 and area<=30:
            c.append(0.6)
        elif area>30 and area<=40:
            c.append(0.6)
        elif area>40 and area<=50:
            c.append(0.45)
        elif area>50 and area<=60:
            c.append(5)
        elif area>60 and area<=70:
            c.append(6)
        elif area>70 and area<=80:
            c.append(6)
        elif area>80 and area<=90:
            c.append(6)
        elif area>90 and area<=100:
            c.append(6)
        elif area>100 and area<=120:
            c.append(6)
        elif area>120 and area<=140:
            c.append(6)
        elif area>140 and area<=160:
            c.append(6)
        elif area>160 and area<=180:
            c.append(6)
        elif area>180 and area<=200:
            c.append(6)
        elif area>200:
            c.append(7)
    raw_data['scale']=c
    raw_data['built']=raw_data['date'].astype(int)/10000-raw_data['built']
    raw_data['built']=raw_data['built'].astype(int)
    raw_data=raw_data[['date','floor','ppa','scale','built','apt_num','psycho_num','price_num']]

    b=[]
    for j in raw_data['date']:
        j=datetime.datetime.strptime(j,'%Y%m%d').date()
        b.append(j)
    start=datetime.date(2010,1,1)
    c=[]
    for i in range(len(b)):
        c.append(int((b[i]-start).days))
    raw_data['date_num']=c

    data=raw_data[['date_num','floor','scale','built']]
    labels=raw_data[['ppa']]
    return data,labels