# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=
import time
import numpy as np
import warnings

import mxnet as mx
from mxnet import gluon, autograd
import gluonnlp as nlp

from SentimentAnalysis.data import get_dataloader, DataTransformer

warnings.filterwarnings('ignore')

from SentimentAnalysis import utils
from SentimentAnalysis.model import TemplatedSentimentNet


def evaluate(net, dataloader, context, log_interval):
    loss = gluon.loss.SigmoidBCELoss()
    total_L = 0.0
    total_sample_num = 0
    total_correct_num = 0
    start_log_interval_time = time.time()
    print('Begin Testing...')

    for i, ((data, valid_length), label) in enumerate(dataloader):
        data = mx.nd.transpose(data.as_in_context(context))
        valid_length = valid_length.as_in_context(context).astype(np.float32)
        label = label.as_in_context(context)

        output = net(data, valid_length)

        L = loss(output, label)
        pred = (output > 0.5).reshape(-1)
        total_L += L.sum().asscalar()
        total_sample_num += label.shape[0]
        total_correct_num += (pred == label).sum().asscalar()

        if (i + 1) % log_interval == 0:
            print('[Batch {}/{}] elapsed {:.2f} s'.format(
                i + 1, len(dataloader),
                time.time() - start_log_interval_time))
            start_log_interval_time = time.time()

    avg_L = total_L / float(total_sample_num)
    acc = total_correct_num / float(total_sample_num)
    return avg_L, acc


def train(net, context, epochs, learning_rate, log_interval):
    trainer = gluon.Trainer(net.collect_params(), 'ftml', {'learning_rate': learning_rate})
    loss = gluon.loss.SigmoidBCELoss()

    # Training/Testing
    for epoch in range(epochs):
        # Epoch training stats
        start_epoch_time = time.time()
        epoch_L = 0.0
        epoch_sent_num = 0
        epoch_wc = 0
        # Log interval training stats
        start_log_interval_time = time.time()
        log_interval_wc = 0
        log_interval_sent_num = 0
        log_interval_L = 0.0

        for i, ((data, length), label) in enumerate(train_dataloader):
            wc = length.sum().asscalar()
            log_interval_wc += wc
            epoch_wc += wc
            log_interval_sent_num += data.shape[1]
            epoch_sent_num += data.shape[1]

            with autograd.record():
                output = net(data.as_in_context(context).T,
                             length.as_in_context(context).astype(np.float32))
                L = loss(output, label.as_in_context(context)).mean()
            L.backward()
            # Update parameter
            trainer.step(1)

            log_interval_L += L.asscalar()
            epoch_L += L.asscalar()
            if (i + 1) % log_interval == 0:
                print(
                    '[Epoch {} Batch {}/{}] elapsed {:.2f} s, '
                    'avg loss {:.6f}, throughput {:.2f}K wps'.format(
                        epoch, i + 1, len(train_dataloader),
                               time.time() - start_log_interval_time,
                               log_interval_L / log_interval_sent_num, log_interval_wc
                               / 1000 / (time.time() - start_log_interval_time)))
                # Clear log interval training stats
                start_log_interval_time = time.time()
                log_interval_wc = 0
                log_interval_sent_num = 0
                log_interval_L = 0

        end_epoch_time = time.time()

        test_avg_L, test_acc = evaluate(net, test_dataloader, context, log_interval)
        print('[Epoch {}] train avg loss {:.6f}, test acc {:.2f}, '
              'test avg loss {:.6f}, throughput {:.2f}K wps'.format(
              epoch, epoch_L / epoch_sent_num, test_acc, test_avg_L,
                   epoch_wc / 1000 / (end_epoch_time - start_epoch_time)))


if __name__ == '__main__':
    options = utils.parse_args()
    context = mx.gpu(options.gpu_index)

    print('Loading pretrained model and vocabulary')

    lm_model, vocab = nlp.model.get_model(name=options.lm_name,
                                          dataset_name=options.lm_dataset,
                                          pretrained=True,
                                          ctx=context)

    print('Loading IMDB dataset')
    train_dataset, test_dataset = [nlp.data.IMDB(root='data/imdb', segment=segment)
                                   for segment in ('train', 'test')]

    transformer = DataTransformer(vocab)
    train_dataset, train_data_lengths = transformer.preprocess_dataset(train_dataset)
    test_dataset, test_data_lengths = transformer.preprocess_dataset(test_dataset)

    train_dataloader, test_dataloader = get_dataloader(train_dataset, test_dataset,
                                                       train_data_lengths, options.batch_size,
                                                       options.bucket_num, options.bucket_ratio)

    net = TemplatedSentimentNet(dropout=options.dropout)
    net.embedding = lm_model.embedding
    net.encoder = lm_model.encoder
    net.output.initialize(mx.init.Xavier(), ctx=context)
    net.hybridize()

    print('Using the following network:')
    print(net)

    train(net, context, options.epochs, options.lr, options.log_interval)

    examples = [
        ['This', 'movie', 'is', 'not', 'that', 'good'],
        ['I', 'was', 'bored', 'to', 'death'],
        ['It', 'was', 'the', 'best', 'experience', 'ever'],
        ['Even', 'my', 'dog', 'is', 'a', 'better', 'actor', 'than', 'that', 'guy']
    ]

    for example in examples:
        pred = net(
            mx.nd.reshape(
                mx.nd.array(vocab[example], ctx=context),
                shape=(-1, 1)), mx.nd.array([4], ctx=context)).sigmoid()

        print('Positiveness of the sentiment "{}" is {}'.format(' '.join(example),
                                                                pred[0].asscalar()))
