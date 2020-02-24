#대부분의 내용은 train.py의 내용과 일치하고 마지막의 train함수가 아닌 새로 작성한 test함수를 사용하였다.

import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.mlp import mlp
from trainers.example_trainer import ExampleTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args


def main():
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    create_dirs([config.summary_dir, config.checkpoint_dir])

    sess = tf.Session()

    data = DataGenerator(config)
    
    model = mlp(config)
    
    logger = Logger(sess, config)
    
    trainer = ExampleTrainer(sess, model, data, config, logger)
    
    model.load(sess)
    #trainer파일을 확인하면 trainer.train()과 새로 작성한 trainer.test()의 차이를 확인할 수 있다.
    #y는 테스트데이터의 실제 ppa, result는 학습된 모델의 추정 ppa값을 리스트로 받아온다.
    #result는 세션의 return이 [1][데이터개수]의 2차원 리스트의 형태이고 [0][i]로 각 input의 결과를 확인할 수 있다
    y,result=trainer.test()
    cnt=0
    print(result[0])
    for i in range(len(y)):
        #실제값-추측값을 실제값으로 나누어 오차10%내의 데이터의 수를 센다
        if(abs(y[i]-float(result[0][i]))/y[i]<=0.1):
            cnt+=1
    print('10% 내외로 예측한 데이터는 ',cnt/len(y),'% 이다')

if __name__ == '__main__':
    main()
