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
import multiprocessing as mp

import gluonnlp as nlp
from mxnet import gluon


class DataTransformer:
    def __init__(self, vocab):
        self._vocab = vocab
        self._tokenizer = nlp.data.SpacyTokenizer('en')
        self._length_clip = nlp.data.ClipSequence(500)

    def preprocess_dataset(self, dataset):
        start = time.time()
        print('Tokenize using spaCy...')
        with mp.Pool() as pool:
            # Each sample is processed in an asynchronous manner.
            dataset = gluon.data.SimpleDataset(pool.map(self._preprocess, dataset))
            lengths = gluon.data.SimpleDataset(pool.map(self._get_length, dataset))
        end = time.time()
        print('Done! Tokenizing Time={:.2f}s, #Sentences={}'.format(end - start, len(dataset)))
        return dataset, lengths

    def _preprocess(self, x):
        data, label = x
        label = int(label > 5)
        # A token index or a list of token indices is
        # returned according to the vocabulary.
        data = self._vocab[self._length_clip(self._tokenizer(data))]
        return data, label

    def _get_length(self, x):
        return float(len(x[0]))


def get_dataloader(train_dataset, test_dataset, train_data_lengths,
                   batch_size, bucket_num, bucket_ratio):
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0, ret_length=True),
        nlp.data.batchify.Stack(dtype='float32'))

    batch_sampler = nlp.data.sampler.FixedBucketSampler(
        train_data_lengths,
        batch_size=batch_size,
        num_buckets=bucket_num,
        ratio=bucket_ratio,
        shuffle=True)

    print(batch_sampler.stats())

    train_dataloader = gluon.data.DataLoader(dataset=train_dataset, batch_sampler=batch_sampler,
                                             batchify_fn=batchify_fn)

    test_dataloader = gluon.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                            shuffle=False, batchify_fn=batchify_fn)

    return train_dataloader, test_dataloader
