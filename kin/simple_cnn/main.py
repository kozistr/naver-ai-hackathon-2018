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

import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
from dataset import KinQueryDataset, preprocess


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


def conv2d(x, f=64, k=1, s=1, pad='SAME', name="conv2d"):
    return tf.layers.conv2d(x,
                            filters=f, kernel_size=k, strides=s,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                            bias_initializer=tf.zeros_initializer(),
                            bias_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                            padding=pad,
                            name=name)


def dense(x, units, name='fc'):
    return tf.layers.dense(x, units,
                           kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                           bias_initializer=tf.zeros_initializer(),
                           bias_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
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


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=151)
    args.add_argument('--batch', type=int, default=64)
    args.add_argument('--strmaxlen', type=int, default=400)
    args.add_argument('--embedding', type=int, default=128)
    args.add_argument('--threshold', type=float, default=0.5)
    args.add_argument('--bn', type=bool, default=False)
    args.add_argument('--lr', type=float, default=1e-4)
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/kin/'

    # model's specification (hyper-parameters)
    input_size = config.embedding * config.strmaxlen
    output_size = 1
    fc_unit = 1024
    conv_filters = 128
    learning_rate = config.lr
    seq_length = config.strmaxlen
    character_size = 251

    x = tf.placeholder(tf.int32, [None, config.strmaxlen])
    y_ = tf.placeholder(tf.float32, [None, output_size])

    # embeddings
    char_embedding = tf.get_variable('char_embedding',
                                     shape=[character_size, config.embedding],
                                     initializer=tf.random_uniform([character_size, config.embedding],
                                                                   minval=-1, maxval=1.)
                                     )
    embedded = tf.nn.embedding_lookup(char_embedding, x)
    embedded = tf.expand_dims(embedded, -1)

    layers = []
    filter_size = [3, 4, 5]
    for idx, fs in enumerate(filter_size):
        with tf.variable_scope('conv_layer-%d' % fs):
            filter_shape = [fs, config.embedding, 1, conv_filters]

            weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=1. / conv_filters), name='W')
            bias = tf.Variable(tf.constant(0., shape=[conv_filters]), name='b')

            x_ = tf.nn.conv2d(embedded, filter=weight, strides=[1, 1, 1, 1], padding='valid', name='conv2d')
            x_ = tf.nn.bias_add(x_, bias)
            x_ = tf.nn.relu(x_)

            x_ = tf.nn.max_pool(x_,
                                ksize=[1, seq_length - conv_filters + 1, 1, 1],
                                strides=[1, 1, 1, 1],
                                padding='valid',
                                name='max_pool2d')
            layers.append(x_)

    layers_concat = tf.concat(layers, axis=3)
    flat = tf.layers.flatten(layers_concat)

    x_ = tf.nn.dropout(flat, .5)

    x_ = dense(x_, fc_unit)`
    x_ = instance_norm(x_)
    x_ = tf.nn.relu(x_)
    x_ = tf.nn.dropout(x_, .5)

    logits = dense(x_, output_size)
    prob = tf.nn.sigmoid(logits)

    # logistic loss
    bce_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y_))

    # Adam Optimizer
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(bce_loss)
    # RMSProp Optimizer
    # train_step = tf.train.RMSPropOptimizer(learning_rate, momentum=0.9).minimize(bce_loss)

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
                                       y_: labels
                                   })

                print('Batch : ', i + 1, '/', one_batch_size, ', BCE in this minibatch: ', float(loss))
                avg_loss += float(loss)

            print('epoch:', epoch, ' train_loss:', float(avg_loss / one_batch_size))

            min_loss = avg_loss

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
