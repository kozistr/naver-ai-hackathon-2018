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


he_normal = tf.keras.initializers.he_normal(),
regularizer = tf.contrib.layers.l2_regularizer(1e-4)


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
        pred = sess.run(prob, feed_dict={x: preprocessed_data})
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
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                            bias_initializer=tf.constant_initializer(0.),
                            bias_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                            padding=pad,
                            name=name)


def dense(x, units, name='fc'):
    return tf.layers.dense(x, units,
                           kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                           bias_initializer=tf.constant_initializer(0.),
                           bias_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                           name=name)


def batch_norm(x, momentum=0.9, eps=1e-5, is_train=True, name="bn"):
    return tf.layers.batch_normalization(inputs=x,
                                         momentum=momentum,
                                         epsilon=eps,
                                         scale=True,
                                         trainable=is_train,
                                         name=name)


def instance_norm(x, name="ibn"):
    epsilon = 1e-9

    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True, name=name)

    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))


class CharCNN(object):
    def __init__(self, sequence_length=400, num_classes=1, vocab_size=251,
                 embedding_size=300, filter_sizes=(1, 2, 3, 4, 5), num_filters=128, lr=5e-4, fc_unit=256):

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        with tf.name_scope("embedding"):
            # self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1., 1.), name="W")
            self.W = tf.get_variable('lookup-W', shape=[vocab_size, embedding_size],
                                     initializer=tf.keras.initializers.he_uniform())
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # print(self.embedded_chars_expanded.get_shape().as_list())  # (batch_size, 400, 300, 1)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%d-%d" % (filter_size, i)):
                conv = conv2d(
                    self.embedded_chars_expanded,
                    num_filters,
                    (filter_size, embedding_size),
                    pad='VALID',
                    name='conv2d-1'
                )
                conv = tf.nn.relu(conv)

                pooled = tf.layers.average_pooling2d(
                    conv,
                    (sequence_length - filter_size + 1, 1),
                    (1, 1),
                    padding='VALID',
                    name='avg_pool2d'
                )

                pooled_outputs.append(pooled)

        # Combine all the pooled features
        self.h_pool = tf.concat(pooled_outputs, 3)
        # print(self.h_pool.get_shape().as_list())  # (batch_size, 1, 1, 512)

        self.h_pool = tf.layers.flatten(self.h_pool)  # (batch_size, 512)

        self.h_drop = tf.nn.dropout(self.h_pool, .5, name='do-0')

        # Final (un-normalized) scores and predictions
        with tf.name_scope("output"):
            fc1 = dense(self.h_drop, fc_unit, name='fc1')
            fc1 = tf.nn.relu(fc1)
            fc1 = tf.nn.dropout(fc1, .5, name='do1')

            self.scores = dense(fc1, output_size, name='fc2')
            self.prob = tf.sigmoid(self.scores)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            self.bce_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores,
                                                                                   labels=self.input_y))
        with tf.name_scope("train"):
            self.train_step = tf.train.AdamOptimizer(lr).minimize(self.bce_loss)


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=51)
    args.add_argument('--batch', type=int, default=64)
    args.add_argument('--strmaxlen', type=int, default=400)
    args.add_argument('--threshold', type=float, default=0.5)
    args.add_argument('--lr', type=float, default=5e-4)
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/kin/'

    # model's specification (hyper-parameters)
    fc_unit = 1024
    conv_filters = 128
    embeddings = 300
    seq_length = config.strmaxlen
    input_size = embeddings * seq_length
    output_size = 1
    learning_rate = config.lr
    character_size = 251

    charcnn = CharCNN(lr=learning_rate)
    x = charcnn.input_x
    y_ = charcnn.input_y
    prob = charcnn.prob
    bce_loss = charcnn.bce_loss
    train_step = charcnn.train_step

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

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

        for epoch in range(config.epochs):
            avg_loss = 0.
            for i, (data, labels) in enumerate(_batch_loader(dataset, config.batch)):
                _, loss = sess.run([train_step, bce_loss],
                                   feed_dict={
                                       x: data,
                                       y_: labels,
                                   })

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
