
from utils import *
from sklearn.utils import shuffle
import time
import utils

import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class LinearRegression:

    def __init__(self, file, lr, batch_size, target, epochs):
        self.learning_rate = lr
        self.batch_size = batch_size
        self.file_name = file
        self.target = target
        self.epochs = epochs

    def build_graph(self):
        self._import_data()
        self._initializer()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

    def _import_data(self):
        self.data = shuffle(read_data(self.file_name))

    def _initializer(self):
        train_data = tf.data.Dataset.from_tensor_slices((self.data.loc[:7000,[self.target]],
                                                        self.data.loc[:7000, ['T']]))

        test_data = tf.data.Dataset.from_tensor_slices((self.data.loc[7000:,[self.target]],
                                                        self.data.loc[7000:,['T']]))
        train_data = train_data.batch(self.batch_size)
        test_data = test_data.batch(self.batch_size)

        ## Create iterator
        iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                   train_data.output_shapes)

        self.target, self.feature = iterator.get_next()

        self.train_init = iterator.make_initializer(train_data)	# initializer for train_data
        self.test_init = iterator.make_initializer(test_data)	# initializer for train_data

    def _create_loss(self):
        self.w = tf.get_variable('weights',shape=(1,1),initializer=tf.random_normal_initializer(0,0.01)
                                 ,dtype=tf.float32)
        self.b = tf.get_variable('biases',shape=(1,1),initializer=tf.zeros_initializer(),dtype=tf.float32)

        ## check the input and outputs of matmul
        target_pred = tf.matmul(self.feature, self.w) + self.b

        loss = tf.square(self.target - target_pred, name='loss')
        self.loss = tf.reduce_mean(loss)

    def _create_optimizer(self):
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss,global_step=global_step)

    def _create_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def train(self):
        ## Check if chepoint is available, else create a trainer
        #utils.safe_mkdir('checkpoints')
        start_time = time.time()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))

            #if ckpt and ckpt.model_checkpoint_path:
            #    saver.restore(sess,ckpt.model_checkpoint_path)
            #else:
            saver = tf.train.Saver()

        ## Create the tensorboard graph object
            writer = tf.summary.FileWriter('graphs/linReg'+str(self.learning_rate),sess.graph)


            sess.run(tf.global_variables_initializer())

            # train the model n_epochs times
            for i in range(self.epochs):
                sess.run(self.train_init)	# drawing samples from train_data
                total_loss = 0
                n_batches = 0
                try:
                    while True:
                        _, l = sess.run([self.optimizer, self.loss])
                        total_loss += l
                        n_batches += 1
                except tf.errors.OutOfRangeError:
                    pass
                print('Avg. loss epoch {0}: {1}'.format(i, total_loss/n_batches))

            loss_batch, summary = sess.run([self.loss,self.summary_op])
            writer.add_summary(summary,global_step=global_step)

            saver.save(sess,'checkpoints/linReg',self.epochs)
            print('Total time: {0} seconds'.format(time.time() - start_time))

        writer.close()

            # test the model
            # sess.run(test_init)			# drawing samples from test_data
            # total_correct_preds = 0
            # try:
            #     while True:
            #         accuracy_batch = sess.run(accuracy)
            #         total_correct_preds += accuracy_batch
            # except tf.errors.OutOfRangeError:
            #     pass
            #
            # print('Accuracy {0}'.format(total_correct_preds/n_test))

## Set hyperparameters
file = "data/AirQualityUCI.xlsx"
lr = 0.1
batch_size = 30
target = 'CO(GT)'
epochs = 10

def main():
    model = LinearRegression(file, lr, batch_size, target, epochs)
    model.build_graph()
    model.train()

if __name__ == '__main__':
    main()
