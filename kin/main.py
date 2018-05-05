# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import os

import numpy as np
import tensorflow as tf
try:
    from tensorflow import keras
    K = keras.backend
    K.set_learning_phase(True)
except ImportError:
    raise ImportError

import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
from dataset import KinQueryDataset, preprocess

he_normal = tf.contrib.layers.variance_scaling_initializer(factor=1., mode='FAN_AVG', uniform=True)
regularizer = tf.contrib.layers.l2_regularizer(1e-2)


# DONOTCHANGE: They are reserved for nsml

# This is for nsml leaderboard
def bind_model(sess, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *args):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *args):
        saver = tf.train.Saver()
        # find checkpoint
        ckpt = tf.train.get_checkpoint_state(dir_name)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(dir_name, checkpoint))
        else:
            raise NotImplemented('No checkpoint!')
        print('Model ', dir_name, ' loaded')

    def infer(raw_data, **kwargs):
        """
        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py 에서 작성한 pre-process 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = preprocess(raw_data, config.strmaxlen)
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        K.set_learning_phase(False)
        pred = sess.run(prob, feed_dict={x: preprocessed_data, do_rate: 0.0})
        clipped = np.array(pred > config.threshold, dtype=np.int)
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        return list(zip(pred.flatten(), clipped.flatten()))

    # DONOTCHANGE: They are reserved for nsml
    # nsml 에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def _batch_loader(iterable, n=1):
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]


def conv2d(x, f, k, s=1, pad='SAME', name="conv2d"):
    return tf.layers.conv2d(x,
                            filters=f, kernel_size=k, strides=s,
                            kernel_initializer=he_normal,
                            kernel_regularizer=regularizer,
                            bias_initializer=tf.constant_initializer(0.),
                            padding=pad,
                            name=name)


def dense(x, units, name='fc'):
    return tf.layers.dense(x, units,
                           kernel_initializer=he_normal,
                           kernel_regularizer=regularizer,
                           bias_initializer=tf.constant_initializer(0.),
                           name=name)


class CharCNN(object):
    def __init__(self, sequence_length=400, num_classes=1, vocab_size=251, embedding_size=300,
                 filter_sizes=(10, 9, 7, 5, 3), num_filters=256, fc_unit=1024, seq_feat_dim=6,
                 n_rnn_layers=2, n_highway_layers=1, num_rnn_cell=64,
                 th=1e-6, batch_size=64):
        # filter_sizes for word-model : (1, 2, 3, 4), 4~ are not good
        # one conv-max layer is enough
        # bigger seq_len and n_filters, more drop rate (0.5~0.8) # batch_norm is not good in this case than drop-out
        # drop out rate 0.7 is the best
        # embeddings size 100 ~ 600, 300 is the best (currently)
        # batch_size : 64 or 128
        # word-level CNN would work better
        # max pooling is better than others, k-max-pool is also good
        # spatial drop-out on the first embed is good
        # VDCNN is not good
        # DPCNN is also not good
        # CharCNN is also not good
        # ~RNN and CNN+RNN are worse than ~CNN
        # At char-level CNN, filter_sizes : 1~10, (but do not use all of them)
        # k_max_pool : k -> 3
        # activation function : ReLU (Threshold ReLU, leaky_ReLE), SeLU ...hmm...
        # gradient norm : clipped at 5. or 7. (5 is better)
        # spatial drop-out is very good

        # phase2
        # /12 : fs = (10, 9, 7, 5, 3) with 400 embeddings, do : 0.7
        # /9  : fs = (10, 9, 7, 5, 3) with 384 embeddings
        # /8  : fs = (10, 9, 7, 5, 3) with max_pool1d -> failed...
        # (will) /5 : fs = (1, 2, 3, 4) with w2v
        # /4  : fs = (10, 9, 7, 5, 3) with 300 embeddings
        # /3  : fs = (10, 9, 7, 5, 3) with 256 embeddings
        # /2  : fs = (3, 4, 5)
        # /1  : fs = (10, 7, 5, 3)

        # phase1
        # /140 : fs = (10, 7, 5, 3) with some changes...
        # /139 : fs = (10, 9, 7, 5, 3) with some changes...
        # /134 : fs = (1 ~ 10)
        # /129 : fs = (10, 7, 5, 3, 2)
        # /128 : fs = (10, 7, 5, 3)  # best

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.do_rate = tf.placeholder(tf.float32, name='dropout-rate')

        with tf.name_scope("embedding"):
            self.sdo = tf.contrib.keras.layers.SpatialDropout1D(self.do_rate)

            self.W = tf.get_variable('lookup-W', shape=[vocab_size, embedding_size],
                                     initializer=he_normal)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars = self.sdo(self.embedded_chars)
            # self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%d-%d" % (filter_size, i)):
                conv = tf.layers.conv1d(
                    self.embedded_chars,
                    num_filters,
                    filter_size,
                    kernel_initializer=he_normal,
                    kernel_regularizer=regularizer,
                    bias_initializer=tf.zeros_initializer(),
                    padding='VALID',
                    name='conv1d'
                )
                """
                conv = tf.layers.conv2d(
                    self.embedded_chars_expanded,
                    num_filters,  # min(50 * filter_size, 200),
                    (filter_size, embedding_size),
                    kernel_initializer=he_normal,
                    kernel_regularizer=regularizer,
                    bias_initializer=tf.zeros_initializer(),
                    padding='VALID',
                    name='conv2d'
                )
                """

                conv = tf.where(tf.less(conv, th), tf.zeros_like(conv), conv)  # ThresholdReLU
                # conv = tf.nn.relu(conv)

                conv = tf.layers.dropout(conv, self.do_rate)

                # k(3)-max-pooling1d
                pooled = tf.nn.top_k(tf.transpose(conv, [0, 2, 1]), k=3, sorted=False)[0]
                pooled = tf.transpose(pooled, [0, 2, 1])

                # max-pooling1d
                """
                pooled = tf.layers.max_pooling1d(
                    inputs=conv,
                    pool_size=3,
                    strides=1,
                    padding='VALID',
                    name='max_pool1d'
                )
                """

                """
                pooled = tf.layers.max_pooling2d(
                    conv,
                    (sequence_length - filter_size + 2 - seq_feat_dim, 1),
                    (1, 1),
                    padding='VALID',
                    name='max_pool2d'
                )
                """

                pooled_outputs.append(pooled)

        # Combine all the pooled features
        # self.h_concat = tf.concat(pooled_outputs, 3)
        self.h_concat = tf.concat(pooled_outputs, 1)
        self.h_pool = tf.layers.flatten(self.h_concat)  # (batch_size, 512/1024)
        # self.h_pool = tf.reshape(self.h_pool, (-1, seq_feat_dim, num_filters * len(filter_sizes)))
        self.h_drop = tf.layers.dropout(self.h_pool, self.do_rate, name='do-0')

        """
        with tf.variable_scope("highway"):
            bias = -2.
            x = self.h_drop
            for i in range(n_highway_layers):
                g = dense(x, x.get_shape()[-1], name='hh-g-dense')
                g = tf.where(tf.less(g, th), tf.zeros_like(g), g)

                t = dense(x, x.get_shape()[-1], name='hh-t-dense') + bias
                t = tf.nn.sigmoid(t)

                x = t * g + (1. - t) * x

        hh_out = tf.reshape(x, (batch_size, embedding_size, -1))
        hh_out = tf.layers.dropout(hh_out, self.do_rate, name='do-1')

        with tf.variable_scope("RNN"):
            rnn_cells = []
            for _ in range(n_rnn_layers):
                cell = tf.nn.rnn_cell.BasicLSTMCell(num_rnn_cell, state_is_tuple=True, forget_bias=0.)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, self.do_rate)
                rnn_cells.append(cell)
            multi_rnn = tf.nn.rnn_cell.MultiRNNCell(rnn_cells, state_is_tuple=True)
            out, _ = tf.nn.dynamic_rnn(multi_rnn, self.h_drop,
                                       dtype=tf.float32)
            out = tf.layers.flatten(out)
        """

        out = self.h_drop

        # Final (un-normalized) scores and predictions
        with tf.name_scope("output"):
            fc1 = dense(out, fc_unit, name='fc1')
            fc1 = tf.where(tf.less(fc1, th), tf.zeros_like(fc1), fc1)
            fc1 = tf.layers.dropout(fc1, self.do_rate, name='do-3')

            self.scores = dense(fc1, output_size, name='fc2')
            self.prob = tf.nn.sigmoid(self.scores)  # scaling to [0, 1]

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            self.bce_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores,
                                                                                   labels=self.input_y))
        # with tf.name_scope("train"):
        #     self.train_step = tf.train.AdamOptimizer(lr).minimize(self.bce_loss)


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--threshold', type=float, default=0.5)
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=100 + 1)
    args.add_argument('--batch', type=int, default=64)
    args.add_argument('--strmaxlen', type=int, default=400)  # 400
    args.add_argument('--embeds', type=int, default=384)  # 256 + 128  # 300
    args.add_argument('--fc_units', type=int, default=1024)
    args.add_argument('--filters', type=int, default=256)
    args.add_argument('--l2_reg', type=float, default=0.)
    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--opt', type=str, default='adam')
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/kin/'

    # model's specification (hyper-parameters)
    seq_length = config.strmaxlen
    embeddings = config.embeds
    input_size = embeddings * seq_length
    output_size = config.output
    character_size = 251
    lr_lower_boundary = 2e-5

    charcnn = CharCNN(embedding_size=embeddings, sequence_length=seq_length, num_filters=config.filters,
                      fc_unit=config.fc_units, batch_size=config.batch)
    x = charcnn.input_x
    y_ = charcnn.input_y
    do_rate = charcnn.do_rate
    prob = charcnn.prob
    bce_loss = charcnn.bce_loss

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)

    # DONOTCHANGE: Reserved for nsml
    bind_model(sess=sess, config=config)

    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        dataset = KinQueryDataset(DATASET_PATH, config.strmaxlen)

        dataset_len = len(dataset)
        one_batch_size = dataset_len // config.batch
        if dataset_len % config.batch != 0:
            one_batch_size += 1

        global_step = tf.Variable(0, name="global_step", trainable=False)
        learning_rate = tf.train.exponential_decay(config.lr, global_step,
                                                   config.epochs * one_batch_size, 0.95, staircase=True)
        lr_update = tf.clip_by_value(learning_rate, 1e-5, 1e-3, name='lr-clipped')
        if config.opt == 'adam':
            optimizer = tf.train.AdamOptimizer(lr_update)  # .minimize(bce_loss)
        elif config.opt == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr_lower_boundary)
        else:
            raise NotImplementedError
        gradients, variables = zip(*optimizer.compute_gradients(bce_loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

        sess.run(tf.global_variables_initializer())

        for epoch in range(config.epochs):
            avg_loss = 0.
            for i, (data, labels) in enumerate(_batch_loader(dataset, config.batch)):
                _, loss, _ = sess.run([train_op, bce_loss, global_step],
                                      feed_dict={
                                          x: data,
                                          y_: labels,
                                          do_rate: 0.7,
                                      })

                _ = tf.train.global_step(sess, global_step)

                print('Batch : ', i + 1, '/', one_batch_size, ', BCE in this minibatch: ', float(loss))
                avg_loss += float(loss)

            print('epoch:', epoch, ' train_loss:', float(avg_loss / one_batch_size))

            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=float(avg_loss / one_batch_size), step=epoch)

            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)

    # [(0.3, 0), (0.7, 1), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            queries = f.readlines()
        res = []
        for batch in _batch_loader(queries, config.batch):
            temp_res = nsml.infer(batch)
            res += temp_res

        print(res)
