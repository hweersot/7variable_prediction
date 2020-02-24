from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class ExampleTrainer(BaseTrain):
    #상속한 클래스의 초기화함수 실행
    def __init__(self, sess, model, data, config,logger):
        super(ExampleTrainer, self).__init__(sess, model, data, config,logger)
    ㄴ
    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        accs = []
        #epoch 마다 batch 별 loss와 정확도 평균계산
        #데이터 만개에 batch_size 10이라면 천개의 batch내의 loss,accuracy들의 평균계산
        for _ in loop:
            #batch 1개 학습 실행
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            'acc': acc,
        }
        #가중치 저장 및 요약파일 write
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        #data_generator의 next_batch함수를 이용해 쪼개진 데이터를 받아와 batch당(16개씩) session 실행
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        #session에 인자로 넣을 dict 미리 생성
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        #feed_dict는 원래의 인자이름이고 같은 이름으로 생성한 {입력16*7줄,출력16*1줄,True}를 인자값으로 대입

        #sess.run()은 기본적으로 sess.run([알고싶은 값1,알고싶은 값2],feed_dict={앞의 값을 구하기 위해 필요한 변수:변수 값})

        #ex) sess.run([y],feed_dict={self.model.x:x})     y=wx+b 일 떄
        _, loss, acc = self.sess.run([self.model.train_step, self.model.cost, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc

    
    def test(self):
        xx, yy = self.data.get_full_dataset()
        #accuracy계산을 위해 result에 예측값list 저장
        result = self.sess.run([self.model.result],
                                     feed_dict={self.model.x: xx, self.model.is_training: False})
        return yy,result
