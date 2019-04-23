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
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU to use')
    parser.add_argument('--embedding_dim', type=int, default=300, help='glove embedding dim')
    parser.add_argument('--logging_path', default="./log_glove_300.txt",
                        help='logging file path')
    parser.add_argument('--model_path', default='./model_glove_300.params',
                        help='saving model in model_path')
    parser.add_argument('--hidden', type=int, default=300, help='hidden units in bilstm')
    parser.add_argument('--lstm_dropout', type=float, default=0.5,
                        help='dropout applied to lstm layers')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=14, help='training epochs')
    parser.add_argument('--seq_len', type=int, default=500, help='max length of sequences')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout applied to fully connected layers')
    return parser.parse_args()
