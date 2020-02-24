#train.py와 test.py 사실상 리모컨과 같이 큰 틀에서의 명령만을 전달하며 세부적인 코드는 각 모듈에서 수행된다.

import tensorflow as tf
from data_loader.data_generator import DataGenerator
from models.mlp import mlp
from trainers.example_trainer import ExampleTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
#텐서플로우와 mlp10 폴더내에서 사용할 소스와 기능들을 호출한다.
#최상위 실행파일인 train.py는 대부분의 기능을 호출한다.


def main():
    #-c 'json파일경로'로 받아온 json경로를 config객체에 저장한다
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # 모델의 학습 결과와 가중치를 저장할 경로를 설정한다
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # 텐서플로우의 세션을 생성한다
    sess = tf.Session()
    # 데이터를 불러온다. 전달한 config객체는 batch사이즈로 데이터를 쪼개기위해 사용된다
    data = DataGenerator(config)
    # 사용할 모델의 개형을 불러온다. 해당 프로젝트에는 input사이즈외에 참고하지 않았지만
    #본래 모델의 깊이,모양,loss함수,optimizer 등 config 값에 따라 다른 모델을 불러올 수 있다
    model = mlp(config)
    # 학습진행과 저장을 담당하는 logger객체를 생성한다
    logger = Logger(sess, config)
    #먼저 생성한 학습에 필요한 세션,모델,데이터셋,설정,logger를 전달해 학습 준비를 마친다
    trainer = ExampleTrainer(sess, model, data, config, logger)
    #기존에 학습중이던 같은 모델이 있다면 해당 모델을 이어서 학습한다
    model.load(sess)
    # here you train your model
    trainer.train()


#아래의 형식은 대부분의 프로젝트에서 실행파일에서 찾을 수 있다.
#__name__은 파이썬의 기본 변수로 default='__main__'이고 간접 실행시 해당 파일이름으로 변경된다.
#아래 조건문의 의미는 이 코드가 다른 파일에서 import train과 같이 호출된것이 아닌 
#python train.py와 같이 최초의 실행명령으로 실행되었으면 main함수를 실행하여라 이다
if __name__ == '__main__':
    main()