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
from argparse import Namespace

from .bert_service import BERTService


def test_bert():
    paragraph = 'Before the foundation can be dug, contractors are typically required to verify and have existing utility lines marked, either by the utilities themselves or through a company specializing in such services. This lessens the likelihood of damage to the existing electrical, water, sewage, phone, and cable facilities, which could cause outages and potentially hazardous situations. During the construction of a building, the municipal building inspector inspects the building periodically to ensure that the construction adheres to the approved plans and the local building code. Once construction is complete and a final inspection has been passed, an occupancy permit may be issued.'

    question = 'Who is required to verify and have existing utility lines marked?'

    params = Namespace()
    params.system_properties = {'gpu_id': None, 'model_dir': '.'}

    service = BERTService()
    service.initialize(params)
    result = service.predict([{'question': question, 'paragraph': paragraph}])
    assert result[0]['predicted'] == 'contractors'
