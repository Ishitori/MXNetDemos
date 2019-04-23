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
"""Main file to run training of keyphrase extraction example."""
import multiprocessing
import mxnet as mx
from mxnet import autograd
from mxnet.gluon import Trainer
from mxnet.gluon.data import DataLoader
from mxnet.gluon.nn import HybridSequential, Embedding, Dropout, Dense
from mxnet.gluon.rnn import LSTM
from mxnet.metric import Accuracy

from gluonnlp import data, Vocab, embedding

from KeyphraseExtraction import utils
from KeyphraseExtraction.dataset import INSPECDataset
from KeyphraseExtraction.data_transformer import DataTransformer


def get_vocab(datasets):
    all_words = [word for dataset in datasets for item in dataset for word in item[0]]
    vocab = Vocab(data.count_tokens(all_words))
    glove = embedding.create('glove', source='glove.6B.' + str(args.embedding_dim) + 'd')
    vocab.set_embedding(glove)
    return vocab


def get_model(vocab_size, embedding_size, hidden_size, dropout_rate, classes=3):
    net = HybridSequential()

    with net.name_scope():
        net.add(Embedding(vocab_size, embedding_size))
        net.add(Dropout(args.dropout))
        net.add(LSTM(hidden_size=hidden_size // 2,
                     num_layers=1,
                     layout='NTC',
                     bidirectional=True,
                     dropout=dropout_rate))
        net.add(Dense(units=classes, flatten=False))

    return net


def run_training(net, trainer, train_dataloader, val_dataloader, epochs, model_path, context):
    loss_fn = mx.gluon.loss.SoftmaxCrossEntropyLoss()

    for e in range(epochs):
        train_acc = Accuracy()
        val_acc = Accuracy()
        train_loss = 0.
        total_items = 0

        for i, (data, valid_length, label) in enumerate(train_dataloader):
            items_per_iteration = data.shape[0]
            total_items += items_per_iteration

            data = data.as_in_context(context)
            label = label.as_in_context(context)

            with autograd.record():
                output = net(data)
                output = output.reshape((-1, 3))
                label = label.reshape((-1, 1))
                loss = loss_fn(output, label)

            loss.backward()
            trainer.step(items_per_iteration)

            train_loss += loss.mean().asscalar()
            train_acc.update(label.flatten(), output.argmax(axis=1).flatten())

        for i, (data, valid_length, label) in enumerate(val_dataloader):
            data = data.as_in_context(context)
            label = label.as_in_context(context)

            output = net(data)
            output = output.reshape((-1, 3))
            val_acc.update(label.reshape(-1, 1).flatten(), output.argmax(axis=1).flatten())

        print("Epoch {}. Current Loss: {:.5f}. Train accuracy: {:.3f}, Validation accuracy: {:.3f}."
              .format(e, train_loss / total_items, train_acc.get()[1], val_acc.get()[1]))

    net.save_parameters(model_path)
    return model_path


def run_evaluation(net, test_dataloader, context):
    correct = 0
    extract = 0
    standard = 0

    for i, (data, valid_length, label) in enumerate(test_dataloader):
        data = data.as_in_context(context)
        label = label.as_in_context(context)

        output = net(data).reshape((-1, 3))

        pred = mx.nd.argmax(output, axis=1).asnumpy().flatten()
        label = label.asnumpy().flatten()

        pred2 = [str(int(x)) for x in pred]
        label2 = [str(x) for x in label]

        predstr = ''.join(pred2).replace('0', ' ').split()
        labelstr = ''.join(label2).replace('0', ' ').split()

        extract += len(predstr)
        standard += len(labelstr)

        i = 0
        while i < len(label):
            if label[i] != 0:
                while i < len(label) and label[i] != 0 and pred[i] == label[i]:
                    i += 1

                if i < len(label) and label[i] == pred[i] == 0 or i == len(label):
                    correct += 1
            i += 1

    precision = 1.0 * correct / extract
    recall = 1.0 * correct / standard
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1, correct, extract, standard


def run_prediction(examples):
    result = []

    for example in examples:
        tokens, valid_length, label = transformer(test_dataset._get_article_words(example), [])
        output = model(mx.nd.reshape(mx.nd.array(tokens, ctx=context), shape=(-1, 1)))
        pred = mx.nd.argmax(output.reshape((-1, 3)), axis=1).asnumpy().flatten()

        all_predictions = []
        kp_started = False

        for i, p in enumerate(pred):
            if p == 1:
                all_predictions.append([i])
                kp_started = True

            if p == 2 and kp_started:
                all_predictions[len(all_predictions) - 1].append(i)

            if p == 0 and kp_started:
                kp_started = False

            if i >= valid_length:
                break

        keyphrases = {' '.join([vocab.idx_to_token[tokens[index]] for index in pred])
                      for pred in all_predictions}
        result.append((example, keyphrases))

    return result


if __name__ == '__main__':
    args = utils.parse_args()
    context = mx.cpu(0) if args.gpu is None else mx.gpu(args.gpu)

    train_dataset = INSPECDataset('train')
    dev_dataset = INSPECDataset('dev')
    test_dataset = INSPECDataset('test')

    vocab = get_vocab([train_dataset, dev_dataset])
    transformer = DataTransformer(vocab, args.seq_len)

    train_dataloader = DataLoader(train_dataset.transform(transformer), batch_size=args.batch_size,
                                  shuffle=True, num_workers=multiprocessing.cpu_count() - 3)
    dev_dataloader = DataLoader(dev_dataset.transform(transformer), batch_size=args.batch_size,
                                shuffle=True, num_workers=multiprocessing.cpu_count() - 3)
    test_dataloader = DataLoader(test_dataset.transform(transformer), batch_size=args.batch_size,
                                 shuffle=True, num_workers=multiprocessing.cpu_count() - 3)

    model = get_model(len(vocab), args.embedding_dim, args.hidden, args.lstm_dropout)
    model.initialize(mx.init.Normal(sigma=0.1), ctx=context)
    model[0].weight.set_data(vocab.embedding.idx_to_vec)

    trainer = Trainer(model.collect_params(), 'adam', {'learning_rate': args.learning_rate})

    best_model_path = run_training(model, trainer, train_dataloader, dev_dataloader,
                                   args.epochs, args.model_path, context)

    model_for_eval = get_model(len(vocab), args.embedding_dim, args.hidden, args.lstm_dropout)
    model_for_eval.load_parameters(best_model_path, ctx=context)

    precision, recall, f1, _, _, _ = run_evaluation(model_for_eval, test_dataloader, context)
    print("Test done. Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(precision, recall, f1))

    # articles from Physics Nature: https://www.nature.com/nphys/articles
    examples = ['Stable coherent terahertz synchrotron radiation from controlled relativistic electron bunches. Relativistic electron bunches used in synchrotron light sources are complex media, in which patterns might form spontaneously. These spatial structures were studied over the past decades for very practical reasons. The patterns, which spontaneously appear during an instability, increase the terahertz radiation power by factors exceeding 10,0001,2. However, their irregularity1,2,3,4,5,6,7 largely prevented applications of this powerful source. Here we show that principles from chaos control theory8,9,10 allow us to generate regular spatio-temporal patterns, stabilizing the emitted terahertz power. Regular unstable solutions are expected to coexist with the undesired irregular solutions, and may thus be controllable using feedback control. We demonstrate the stabilization of such regular solutions in the Synchrotron SOLEIL storage ring. Operation of these controlled unstable solutions enables new designs of high-charge and stable synchrotron radiation sources.',
                'Large and reversible myosin-dependent forces in rigidity sensing. Cells sense the rigidity of their environment through localized pinching, which occurs when myosin molecular motors generate contractions within actin filaments anchoring the cell to its surroundings. We present high-resolution experiments performed on these elementary contractile units in cells. Our experimental results challenge the current understanding of molecular motor force generation. Surprisingly, bipolar myosin filaments generate much larger forces per motor than measured in single-molecule experiments. Furthermore, contraction to a fixed distance, followed by relaxation at the same rate, is observed over a wide range of matrix rigidities. Finally, stepwise displacements of the matrix contacts are apparent during both contraction and relaxation. Building on a generic two-state model of molecular motor collections, we interpret these unexpected observations as spontaneously emerging features of a collective motor behaviour. Our approach explains why, in the cellular context, collections of resilient and slow motors contract in a stepwise fashion while collections of weak and fast motors do not. We thus rationalize the specificity of motor contractions implied in rigidity sensing compared to previous in vitro observations.',
                'Conformational control of mechanical networks. Understanding conformational change is crucial for programming and controlling the function of many mechanobiological and mechanical systems such as robots, enzymes and tunable metamaterials. These systems are often modelled as constituent nodes (for example, joints or amino acids) whose motion is restricted by edges (for example, limbs or bonds) to yield functionally useful coordinated motions (for example, walking or allosteric regulation). However, the design of desired functions is made difficult by the complex dependence of these coordinated motions on the connectivity of edges. Here, we develop simple mathematical principles to design mechanical systems that achieve any desired infinitesimal or finite coordinated motion. We specifically study mechanical networks of two- and three-dimensional frames composed of nodes connected by freely rotating rods and springs. We first develop simple principles that govern all networks with an arbitrarily specified motion as the sole zero-energy mode. We then extend these principles to characterize networks that yield multiple specified zero modes, generate pre-stress stability and display branched motions. By coupling individual modules, we design networks with negative Poisson’s ratio and allosteric response. Finally, we extend our framework to networks with arbitrarily specifiable initial and final positions to design energy minima at desired geometric configurations, and create networks demonstrating tristability and cooperativity.']

    for example, prediction in run_prediction(examples):
        print('Keyphrases for "{}"'.format(example))
        print(prediction)

