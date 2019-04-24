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
import ast
import os

import mxnet as mx
from mxnet import nd
import gluonnlp as nlp

from data_pipeline import BERTTokenizer, SQuADTransform
from inference import Result, predictions
from model import BertForQA
from mxnet.gluon.data import SimpleDataset


class BERTService:
    """BERT Service file. It is an entry point for MXNet Model Service call"""
    def __init__(self):
        self.is_initialized = False

        self._ctx = None
        self._net = None
        self._tokenizer = None
        self._data_transformer = None
        self._param_filename = "net_parameters.params"
        self._dev_dataset = None
        self._features = None

    def initialize(self, params):
        gpu_id = params.system_properties.get("gpu_id")
        model_dir = params.system_properties.get("model_dir")

        self._ctx = mx.cpu() if gpu_id is None else mx.gpu(gpu_id)

        bert, vocab = nlp.model.get_model(
            name='bert_12_768_12',
            dataset_name='book_corpus_wiki_en_uncased',
            pretrained=False,
            ctx=self._ctx,
            use_pooler=False,
            use_decoder=False,
            use_classifier=False)

        bert.encoder._output_attention = True
        self._net = BertForQA(bert)
        self._tokenizer = BERTTokenizer(vocab)
        self._data_transformer = SQuADTransform(self._tokenizer, is_training=False)

        param_file_path = os.path.join(model_dir, self._param_filename)

        if not os.path.isfile(param_file_path):
            raise OSError("Parameter file not found {}".format(param_file_path))

        self._net.load_parameters(param_file_path)
        self.is_initialized = True

    def preprocess(self, data):
        if data[0].get('data') is not None:
            data = ast.literal_eval(data[0].get('data').decode('utf-8'))

        paragraph = data[0].get('paragraph')
        question = data[0].get('question')

        # format is: example_id, qas_id, question_text, paragraph_text,
        #            orig_answer_text, answer_offset, is_impossible
        examples, features = self._data_transformer((0, 0, question, paragraph, [''], [0], 0))

        # features are: example_id, input_ids, segment_ids,
        #               valid_length, start_position, end_position
        self._dev_dataset = SimpleDataset(examples)
        self._features = SimpleDataset(features)

        return self._features

    def inference(self, data):
        record = data[0]
        input_ids = nd.array(record[1]).astype('float32').expand_dims(axis=0).as_in_context(self._ctx)
        token_types = nd.array(record[2]).astype('float32').expand_dims(axis=0).as_in_context(self._ctx)
        valid_length = nd.array([record[3]]).astype('float32').as_in_context(self._ctx)

        out = self._net(input_ids, token_types, valid_length)
        output = nd.split(out, axis=2, num_outputs=2)

        start_logits = output[0].reshape((0, -3)).asnumpy()
        end_logits = output[1].reshape((0, -3)).asnumpy()
        all_possible_results = [Result(start.tolist(), end.tolist())
                                for start, end in zip(start_logits, end_logits)]

        return all_possible_results

    def postprocess(self, data):
        """Find answer in the text based on prediction"""
        all_predictions, all_nbest_json, scores_diff_json = predictions(self._dev_dataset,
                                                                        data,
                                                                        self._tokenizer)

        if len(all_nbest_json) == 0 or len(all_nbest_json[0]) == 0:
            return [{'predicted': '',
                     'confidence': 0}]

        return [{'predicted': all_nbest_json[0][0]['text'],
                 'confidence': all_nbest_json[0][0]['probability']}]

    def predict(self, data):
        data = self.preprocess(data)
        data = self.inference(data)
        return self.postprocess(data)


service = BERTService()


def service_inference(data, context):
    result = ""

    if not service.is_initialized:
        service.initialize(context)

    if data is not None:
        result = service.predict(data)

    return result
