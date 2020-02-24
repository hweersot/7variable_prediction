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
    raw_data['built']=raw_data['건축년도']
    raw_data['built']=raw_data['date'].astype(int)/10000-raw_data['built']
    raw_data['built']=raw_data['built'].astype(int)
    raw_data=raw_data.drop(raw_data[['건축년도']],axis=1)
    #column 이름 재설정, 사용할 column만 남김
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
    #3가지 지수 중 데이터가 없어 0으로 지정된 row를 모두 제거한다
    raw_data=raw_data[raw_data.apt_num!=0]
    raw_data=raw_data[raw_data.price_num!=0]
    raw_data=raw_data[raw_data.psycho_num!=0]
    #필요한 column에 0~1 정규화를 수행한다
    scaler = MinMaxScaler(feature_range=(0, 1))
    data=raw_data[['date_num','built','apt_num','psycho_num','price_num','floor_ratio','scale_ratio']]
    labels=raw_data[['ppa']]
    return data,labels


class DataGenerator:
    #self란 DataGenerator클래스를 이용해 만든 객체 자신을 가리킨다.
    def __init__(self, config):
        self.config = config
        # 상기의 csv_read를 사용하여 input과 정답(ppa)를 각각의 리스트에 저장한다   r''는 인코딩과정의 오류가 발생해 해결한 부분이다
        inn,y=csv_read(r'C:\Users\00001234\mlp10\data_loader\bundang_ratio_ver.csv',encoding='Ansi')
        #self.변수명을 아래와 같이 초기화 하면 train.py에서 객체이름.변수명으로 저장된 데이터를 사용가능하다.
        #ex)data.input
        #또한 ExampleTrainer(sess, model, data, config, logger)와 같이 다른 객체를 생성할 때 아래에서 저장한 input과 y값을 사용가능하며 사용법은 마찬가지로 data.input의 형식이다

        #dataframe을 numpy array로 변환한다
        self.input = np.array(inn.values)
        self.y = np.array(y.values)


#input과 y를 1개씩 비교하지 않고 batch크기 씩 비교하기떄문에 [batch_size][input_size]  [batch_size][label_size]를 요청 받을때 마다 반환한다.

    def next_batch(self, batch_size):
#        idx = np.random.choice(500, batch_size)
        #데이터 개수가 6만개라면 6만 미만의 숫자들 중 batch_size(16개) 만큼의 랜덤 숫자를 idx에 저장한다.
        #replace인자가 False일 시 중복된 선택이 없게 한다
        idx = np.random.choice(len(self.input), batch_size,replace=False)

        #return과 달리 yield는 next() 함수를 통해 다른 소스파일에서도 반복문과 같은 기능을 사용할수 있다    ->  example_trainer.py 30번째 줄
        yield self.input[idx], self.y[idx]

#test 시 정확도 측정을 위해 batch 없이 전체 데이터를 반환한다.
    def get_full_dataset(self):
        return self.input, self.y
