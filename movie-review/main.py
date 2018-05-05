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
import torch
import torch.nn.functional as F

from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader

import nsml
from dataset import MovieReviewDataset, preprocess
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML


# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(model, config):
    def save(filename, *args):
        checkpoint = {
            'model': model.state_dict()
        }
        torch.save(checkpoint, filename)

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(filename, *args):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model'])
        print('Model loaded')

    def infer(raw_data, **kwargs):
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = preprocess(raw_data, config.strmaxlen)
        model.eval()

        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        output_prediction = model(preprocessed_data)
        point = output_prediction.data.squeeze(dim=1).tolist()

        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(confidence interval, 포인트)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다.
        # 리더보드 결과에 confidence interval의 값은 영향을 미치지 않습니다.

        return list(zip(np.zeros(len(point)), point))

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)


def collate_fn(data: list):
    review = []
    label = []
    for datum in data:
        review.append(datum[0])
        label.append(datum[1])

    return review, np.array(label)


class Regression(nn.Module):

    def __init__(self, embedding_dim: int, max_length: int):
        super(Regression, self).__init__()

        self.embedding_dim = embedding_dim
        self.character_size = 251
        self.output_dim = 1  # Regression # maybe 10 for cross-entropy would be better than sigmoid method
        self.max_length = max_length
        self.k_sizes = (3, 5, 7)  # (3, 5, 7), (3, 4, 5) best for this DataSet
        self.conv_filter = 256
        self.fc_unit = 512
        self.drop_out = .6  # best for this DataSet

        self.embeddings = nn.ModuleList([nn.Embedding(self.character_size, self.embedding_dim) for _ in range(3)])
        # self.dos = nn.ModuleList([nn.Dropout(self.drop_out) for _ in range(3)])  # would be better
        # self.embeddings = nn.Embedding(self.character_size, self.embedding_dim)  # (251, embeds)

        self.convs = nn.ModuleList([nn.Conv2d(1, self.conv_filter, (ks, self.embedding_dim)) for ks in self.k_sizes])
        # self.convs = nn.ModuleList([nn.Conv1d(self.embedding_dim, self.conv_filter, ks) for ks in self.k_sizes])
        for c in self.convs:
            torch.nn.init.xavier_normal(c.weight)
            torch.nn.init.constant(c.bias, 0.)

        self.do1 = nn.Dropout(self.drop_out)
        self.do2 = nn.Dropout(self.drop_out)

        # self.fc1 = nn.Linear(len(self.k_sizes) * self.conv_filter, self.output_dim)
        # torch.nn.init.xavier_uniform(self.fc1.weight)
        # torch.nn.init.constant(self.fc1.bias, 0.)

        self.fc1 = nn.Linear(len(self.k_sizes) * self.conv_filter, self.fc_unit)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        torch.nn.init.constant(self.fc1.bias, 0.)

        self.fc2 = nn.Linear(self.fc_unit, self.output_dim)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        torch.nn.init.constant(self.fc2.bias, 0.)

    def forward(self, data: list):
        def k_max_pooling(x, dim, k):
            index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
            return x.gather(dim, index)

        batch_size = len(data)

        data_in_torch = Variable(torch.from_numpy(np.array(data)).long()).cuda()

        x_e = []
        for e in self.embeddings:
            x_e.append(e(data_in_torch).unsqueeze(1))

        # x_ = self.embeddings(data_in_torch)  # (batch_size, max_length, embeds)
        # Add SpatialDropout1D
        # x_ = x_.unsqueeze(1)

        x_ = [F.relu(c(x_e[idx])).squeeze(3) for idx, c in enumerate(self.convs)]
        x_ = [F.max_pool1d(l, l.size(2)).squeeze(2) for l in x_]  # k_max_pooling(l, 2, 3)

        x_ = torch.cat(x_, 1)
        x_ = x_.view(batch_size, -1)  # apply self-attention instead of flatten
        flat = self.do1(x_)

        fc1 = self.fc1(flat)
        fc1 = F.relu(fc1)
        fc1 = self.do2(fc1)

        fc2 = self.fc2(fc1)
        output = torch.sigmoid(fc2) * 9 + 1  # scaling (1 ~ 10) # will be changed...

        # output = F.softmax(fc2, dim=1)
        # output = torch.topk(output, 1)[1] + 1  # scaling (1 ~ 10)

        # .type(torch.cuda.FloatTensor)
        # return Variable(output.data)
        return output


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=25 + 1)
    args.add_argument('--batch', type=int, default=128)  # 256
    args.add_argument('--strmaxlen', type=int, default=200)
    args.add_argument('--embedding', type=int, default=128)
    args.add_argument('--lr', type=float, default=2e-4)  # LR Scheduling is needed
    args.add_argument('--flr', type=float, default=2e-5)  # Fine-LR Scheduling is needed
    args.add_argument('--opt', type=str, default='adam')
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/movie_review/'

    model = Regression(config.embedding, config.strmaxlen).cuda()

    # DONOTCHANGE: Reserved for nsml use
    bind_model(model, config)

    criterion = nn.MSELoss()
    if config.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.opt == 'sgd':  # for fine-tuning
        optimizer = optim.SGD(model.parameters(), lr=config.flr, nesterov=True, momentum=.9)
    else:
        raise NotImplementedError

    # DONOTCHANGE: They are reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        dataset = MovieReviewDataset(DATASET_PATH, config.strmaxlen)
        train_loader = DataLoader(dataset=dataset,
                                  batch_size=config.batch,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  num_workers=2)
        total_batch = len(train_loader)

        for epoch in range(config.epochs):
            avg_loss = 0.
            for i, (data, labels) in enumerate(train_loader):
                predictions = model(data)
                label_vars = Variable(torch.from_numpy(labels)).cuda()
                loss = criterion(predictions, label_vars)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('Batch : ', i + 1, '/', total_batch, ', MSE in this minibatch: ', loss.data[0])
                avg_loss += loss.data[0]

            print('epoch:', epoch, ' train_loss:', float(avg_loss / total_batch))
            # nsml ps, 혹은 웹 상의 텐서보드에 나타나는 값을 리포트하는 함수입니다.
            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=float(avg_loss / total_batch), step=epoch)

            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.0, 9.045), (0.0, 5.91), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()
        res = nsml.infer(reviews)

        print(res)
