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
        pred = sess.run(prob, feed_dict={x: preprocessed_data, is_train: False})
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


def Convolutional_Block(inputs, shortcut, num_filters, name, is_training):
    with tf.variable_scope("conv_block_" + str(num_filters) + "_" + name):
        for i in range(2):
            with tf.variable_scope("conv1d_%s" % str(i)):
                filter_shape = [3, inputs.get_shape()[2], num_filters]
                W = tf.get_variable(name='W', shape=filter_shape,
                                    initializer=he_normal,
                                    regularizer=regularizer)
                inputs = tf.nn.conv1d(inputs, W, stride=1, padding="SAME")
                inputs = tf.layers.batch_normalization(inputs=inputs, momentum=0.9, epsilon=1e-5,
                                                       center=True, scale=True, training=is_training)
                inputs = tf.nn.relu(inputs)

    if shortcut is not None:
        return inputs + shortcut

    return inputs


def downsampling(inputs, downsampling_type, name, optional_shortcut=False, shortcut=None):
    if downsampling_type == 'k-maxpool':
        k = math.ceil(int(inputs.get_shape()[1]) / 2)
        pool = tf.nn.top_k(tf.transpose(inputs, [0, 2, 1]), k=k, name=name, sorted=False)[0]
        pool = tf.transpose(pool, [0, 2, 1])
    elif downsampling_type == 'linear':
        pool = tf.layers.conv1d(inputs=inputs, filters=inputs.get_shape()[2], kernel_size=3,
                                strides=2, padding='same', use_bias=False)
    else:  # best
        pool = tf.layers.max_pooling1d(inputs=inputs, pool_size=3, strides=2, padding='same', name=name)

    if optional_shortcut:
        shortcut = tf.layers.conv1d(inputs=shortcut, filters=shortcut.get_shape()[2], kernel_size=1,
                                    strides=2, padding='same', use_bias=False)
        pool += shortcut

    pool = fixed_padding(inputs=pool)
    return tf.layers.conv1d(inputs=pool, filters=pool.get_shape()[2] * 2, kernel_size=1,
                            strides=1, padding='valid', use_bias=False)


def fixed_padding(inputs, kernel_size=3):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [0, 0]])
    return padded_inputs


class TextCNN(object):
    def __init__(self, num_classes=1, sequence_max_length=1012, vocab_size=251, embedding_size=16,
                 depth=29, downsampling_type='maxpool', optional_shortcut=True, lr=1e-3, fc_units=2048):

        self.input_x = tf.placeholder(tf.int32, [None, sequence_max_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        if depth == 9:
            num_layers = [2, 2, 2, 2]
        elif depth == 17:
            num_layers = [4, 4, 4, 4]
        elif depth == 29:  # best
            num_layers = [10, 10, 4, 4]
        elif depth == 49:  # not good
            num_layers = [16, 16, 10, 6]
        else:
            raise NotImplementedError

        with tf.name_scope("embedding"):
            # self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1., 1.), name="W")
            self.W = tf.get_variable('lookup-W', shape=[vocab_size, embedding_size],
                                     initializer=he_normal)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

        self.layers = []

        with tf.variable_scope('temp-conv'):
            filter_shape = [3, embedding_size, 64]
            W = tf.get_variable(name='W_1', shape=filter_shape,
                                initializer=he_normal,
                                regularizer=regularizer)
            inputs = tf.nn.conv1d(self.embedded_chars, W, stride=1, padding="SAME")

        self.layers.append(inputs)

        for i in range(num_layers[0]):
            if i < num_layers[0] - 1 and optional_shortcut:
                shortcut = self.layers[-1]
            else:
                shortcut = None
            conv_block = Convolutional_Block(inputs=self.layers[-1], shortcut=shortcut, num_filters=64,
                                             is_training=self.is_training, name=str(i + 1))
            self.layers.append(conv_block)
        pool1 = downsampling(self.layers[-1], downsampling_type=downsampling_type, name='pool1',
                             optional_shortcut=optional_shortcut, shortcut=self.layers[-2])
        self.layers.append(pool1)

        for i in range(num_layers[1]):
            if i < num_layers[1] - 1 and optional_shortcut:
                shortcut = self.layers[-1]
            else:
                shortcut = None
            conv_block = Convolutional_Block(inputs=self.layers[-1], shortcut=shortcut, num_filters=128,
                                             is_training=self.is_training, name=str(i + 1))
            self.layers.append(conv_block)
        pool2 = downsampling(self.layers[-1], downsampling_type=downsampling_type, name='pool2',
                             optional_shortcut=optional_shortcut, shortcut=self.layers[-2])
        self.layers.append(pool2)

        for i in range(num_layers[2]):
            if i < num_layers[2] - 1 and optional_shortcut:
                shortcut = self.layers[-1]
            else:
                shortcut = None
            conv_block = Convolutional_Block(inputs=self.layers[-1], shortcut=shortcut, num_filters=256,
                                             is_training=self.is_training, name=str(i + 1))
            self.layers.append(conv_block)
        pool3 = downsampling(self.layers[-1], downsampling_type=downsampling_type, name='pool3',
                             optional_shortcut=optional_shortcut, shortcut=self.layers[-2])
        self.layers.append(pool3)

        for i in range(num_layers[3]):
            if i < num_layers[3] - 1 and optional_shortcut:
                shortcut = self.layers[-1]
            else:
                shortcut = None
            conv_block = Convolutional_Block(inputs=self.layers[-1], shortcut=shortcut, num_filters=512,
                                             is_training=self.is_training, name=str(i + 1))
            self.layers.append(conv_block)

        # Extract 8 most features as mentioned in paper
        self.k_pooled = tf.nn.top_k(tf.transpose(self.layers[-1], [0, 2, 1]), k=8, name='k_pool', sorted=False)[0]

        self.flatten = tf.layers.flatten(self.k_pooled)  # tf.reshape(self.k_pooled, (-1, 512 * 8))

        # Final (un-normalized) scores and predictions
        with tf.variable_scope('fc1'):
            w = tf.get_variable('w', [self.flatten.get_shape()[1], fc_units],
                                initializer=he_normal, regularizer=regularizer)
            b = tf.get_variable('b', [fc_units], initializer=tf.constant_initializer(.1))
            out = tf.matmul(self.flatten, w) + b
            self.fc1 = tf.nn.relu(out)

        # fc2
        with tf.variable_scope('fc2'):
            w = tf.get_variable('w', [self.fc1.get_shape()[1], num_classes],
                                initializer=he_normal, regularizer=regularizer)
            b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(.1))

            self.scores = tf.matmul(self.fc1, w) + b
            self.prob = tf.sigmoid(self.scores)

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
    # model's specification (hyper-parameters)
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=50 + 1)
    args.add_argument('--batch', type=int, default=128)  # best
    args.add_argument('--embeds', type=int, default=16)  # best
    args.add_argument('--depth', type=int, default=29)   # best
    args.add_argument('--shortcut', type=bool, default=True)  # best
    args.add_argument('--strmaxlen', type=int, default=1024)  #
    args.add_argument('--fc_units', type=int, default=1024)   #
    args.add_argument('--threshold', type=float, default=0.5)
    args.add_argument('--lr', type=float, default=9e-3)
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/kin/'

    character_size = 251
    # VDCNN Model
    textcnn = TextCNN(depth=config.depth,
                      embedding_size=config.embeds, sequence_max_length=config.strmaxlen,
                      optional_shortcut=config.shortcut,
                      fc_units=config.fc_units,
                      lr=config.lr)
    x = textcnn.input_x
    y_ = textcnn.input_y
    is_train = textcnn.is_training
    prob = textcnn.prob
    bce_loss = textcnn.bce_loss
    # train_step = textcnn.train_step

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
                                                   config.epochs * one_batch_size, 0.95, staircase=True)
        # train_step = tf.train.AdamOptimizer(learning_rate).minimize(bce_loss)
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        gradients, variables = zip(*optimizer.compute_gradients(bce_loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 7.0)
        train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

        sess.run(tf.global_variables_initializer())

        for epoch in range(config.epochs):
            avg_loss = 0.
            for i, (data, labels) in enumerate(_batch_loader(dataset, config.batch)):
                _, loss, _ = sess.run([train_op, bce_loss, global_step],
                                      feed_dict={
                                          x: data,
                                          y_: labels,
                                          is_train: True,
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
