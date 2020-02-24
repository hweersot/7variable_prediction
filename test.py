import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.mlp import mlp
from trainers.example_trainer import ExampleTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    data = DataGenerator(config)
    # create an instance of the model you want
    model = mlp(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = ExampleTrainer(sess, model, data, config, logger)
    #load model if exists
    model.load(sess)
    # here you train your model
    y,result=trainer.test()
    cnt=0
    print(result[0])
    for i in range(len(y)):
        if(abs(y[i]-float(result[0][i]))/y[i]<=0.1):
            cnt+=1
    print('10% 내외로 예측한 데이터는 ',cnt/len(y),'% 이다')

if __name__ == '__main__':
    main()
