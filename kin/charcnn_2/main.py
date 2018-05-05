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
import math
import os

import numpy as np
import tensorflow as tf

import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
from dataset import KinQueryDataset, preprocess


he_normal = tf.contrib.layers.variance_scaling_initializer(factor=1., mode='FAN_AVG', uniform=True)
regularizer = tf.contrib.layers.l2_regularizer(5e-4)


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
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """
        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = preprocess(raw_data, config.strmaxlen)
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        pred = sess.run(prob, feed_dict={x: preprocessed_data, do_rate: 0.0})
        clipped = np.array(pred > config.threshold, dtype=np.int)
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        return list(zip(pred.flatten(), clipped.flatten()))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
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
                 filter_sizes=(7, 7, 3, 3, 3, 3), num_filters=256, fc_unit=1024,
                 th=1e-6):

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.do_rate = tf.placeholder(tf.float32, name='dropout-rate')

        with tf.name_scope("embedding"):
            self.W = tf.get_variable('lookup-W', shape=[vocab_size, embedding_size],
                                     initializer=he_normal)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        x = self.embedded_chars_expanded

        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%d-%d" % (filter_size, i)):
                x = tf.layers.conv2d(
                    x,
                    num_filters,  # min(50 * filter_size, 200),
                    (filter_size, embedding_size),
                    kernel_initializer=he_normal,
                    kernel_regularizer=regularizer,
                    bias_initializer=tf.zeros_initializer(),
                    padding='VALID',
                    name='conv2d-1'
                )
                x = tf.where(tf.less(x, th), tf.zeros_like(x), x)  # ThresholdReLU
                # x = tf.layers.dropout(x, self.do_rate)

                if i < 2 or i > 4:
                    x = tf.nn.max_pool(x, ksize=[1, 3, 1, 1], strides=[1, 3, 1, 1], padding='VALID', name='pool-%d' % i)

                x = tf.transpose(x, [0, 1, 3, 2], name='trans-%d' % i)

        self.h_pool = tf.layers.flatten(x)
        self.h_drop = tf.layers.dropout(self.h_pool, self.do_rate, name='do-0')

        out = self.h_drop

        # Final (un-normalized) scores and predictions
        with tf.name_scope("output"):
            fc1 = dense(out, fc_unit, name='fc1')
            fc1 = tf.where(tf.less(fc1, th), tf.zeros_like(fc1), fc1)
            fc1 = tf.layers.dropout(fc1, self.do_rate, name='do-3')

            fc2 = dense(fc1, fc_unit, name='fc2')
            fc2 = tf.where(tf.less(fc2, th), tf.zeros_like(fc2), fc2)
            fc2 = tf.layers.dropout(fc2, self.do_rate, name='do-4')

            self.scores = dense(fc2, output_size, name='fc2')
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
    args.add_argument('--epochs', type=int, default=150 + 1)
    args.add_argument('--batch', type=int, default=128)
    args.add_argument('--strmaxlen', type=int, default=1014)
    args.add_argument('--embeds', type=int, default=300)
    args.add_argument('--fc_units', type=int, default=1024)
    args.add_argument('--filters', type=int, default=256)
    args.add_argument('--lr', type=float, default=5e-3)
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/kin/'

    # model's specification (hyper-parameters)
    seq_length = config.strmaxlen
    embeddings = config.embeds
    input_size = embeddings * seq_length
    output_size = config.output
    character_size = 251

    charcnn = CharCNN(embedding_size=embeddings, sequence_length=seq_length, num_filters=config.filters,
                      fc_unit=config.fc_units)
    x = charcnn.input_x
    y_ = charcnn.input_y
    do_rate = charcnn.do_rate
    prob = charcnn.prob
    bce_loss = charcnn.bce_loss
    # train_step = charcnn.train_step

    sess = tf.Session()

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
                                                   config.epochs * one_batch_size, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)  # .minimize(bce_loss)
        gradients, variables = zip(*optimizer.compute_gradients(bce_loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 400.0)
        train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

        sess.run(tf.global_variables_initializer())

        for epoch in range(config.epochs):
            avg_loss = 0.
            for i, (data, labels) in enumerate(_batch_loader(dataset, config.batch)):
                _, loss, _ = sess.run([train_op, bce_loss, global_step],
                                      feed_dict={
                                          x: data,
                                          y_: labels,
                                          do_rate: 0.5,
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
