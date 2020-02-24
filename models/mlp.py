from base.base_model import BaseModel
import tensorflow as tf

#base폴더의 baseModel의 기능을 기본으로 가진 클래스 생성
class mlp(BaseModel):
    def __init__(self, config):
        #BaseModel의 함수의 __init__함수를 호출해 초기화를 수행한다.
        #__init__를 오버라이딩 했기떄문에 super를 통해 호출
        super(mlp, self).__init__(config)
        #아래 구현된 함수 호출
        self.build_model()
        self.init_saver()

    def build_model(self):
        #본격적인 모델 짜올리기 과정
        
        #본래 is_training의 true,false를 통해 학습중인지 테스트 중인지 구분
        self.is_training = tf.placeholder(tf.bool)

        #input과 레이블을 대입할 빈 변수(placeholder) 생성
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None, 1])
        # 모델의 개형 설정      다음은 input->10->10->10->output 의 fully-connected 레이어이다.
        #tf.layer.dense를 통해 데이터타입, 퍼셉트론의 갯수 활성화함수를 설정 후 레이어를 생성할 수 있다
        d1 = tf.layers.dense(self.x, 10, activation=tf.nn.relu, name="dense1")
        d2 = tf.layers.dense(d1, 10, activation=tf.nn.relu, name="dense2")
        d3 = tf.layers.dense(d2, 10, activation=tf.nn.relu, name="dense3")
        d4 = tf.layers.dense(d3, 1, name="dense4")
        #tf.name_scope는 이름 설정에 관여하는 함수이다.
        #아래의 설정된 cost,result 등의 텐서들은 loss라는 카테고리 안에 속하게 되고 후에 호출이나 가시화를 할때 loss로 호출,분류가 가능하다
        with tf.name_scope("loss"):
            #본 모델의 비용함수는 실제값과 예측값 간의 RMSE 결과이다.
            self.cost = tf.reduce_mean(tf.square(d4-self.y))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cost,
                                                                                         global_step=self.global_step_tensor)

#                learning_rate decy 사용 시 코드
#                learning_rate = tf.train.exponential_decay(self.config.learning_rate, self.global_step_tensor,
#                                           200, 0.97, staircase=True)
#                self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cost,
#                                                                                         global_step=self.global_step_tensor)

            correct_prediction = tf.reduce_mean(tf.square(d4-self.y))
            self.result= d4
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

