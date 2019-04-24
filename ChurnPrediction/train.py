import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import Trainer
from mxnet.gluon.data import DataLoader
from mxnet.metric import F1

from ChurnPrediction import utils
from ChurnPrediction.data import get_data_frame, TelcoDataset
from ChurnPrediction.model import TelcoModel


def train(net, train_dataloader, val_dataloader, epochs, context):
    max_f1 = 0
    best_epoch = -1
    trainer = Trainer(net.collect_params(), 'ftml', {'learning_rate': options.lr})
    trainer = Trainer(net.collect_params(), 'ftml', {'learning_rate': options.lr})
    l2_loss_fn = gluon.loss.L2Loss()
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

    for e in range(epochs):
        cum_loss = 0
        total_items = 0

        for i, data in enumerate(train_dataloader):
            total_items += data[0].shape[0]

            for idx in range(0, len(data)):
                data[idx] = data[idx].astype(np.float32).reshape((-1, 1)).as_in_context(context)

            with autograd.record():
                output, decoded = net(*data[:-1])
                l2_loss = l2_loss_fn(decoded, mx.nd.concat(*data[:-1], dim=1))
                classification_loss = loss_fn(output, data[len(data) - 1])
                loss = l2_loss + classification_loss
            loss.backward()
            trainer.step(1)
            cum_loss += loss.mean().asscalar()

        train_f1 = evaluate(net, train_dataloader, context)
        val_f1 = evaluate(net, val_dataloader, context)

        print('Epoch [{}]: Train F1 {:.3f}, Val F1 {:.3f}. Train loss {:.6f}'
              .format(e, train_f1, val_f1, cum_loss / total_items))

        if val_f1 > max_f1:
            net.save_parameters('best_model.params')
            max_f1 = val_f1
            best_epoch = e

    print('Best model found on epoch {}, Val F1 {:.3f}'.format(best_epoch, max_f1))


def evaluate(net, dataloader, context):
    f1 = F1()

    for i, data in enumerate(dataloader):
        for idx in range(0, len(data)):
            data[idx] = data[idx].astype(np.float32).reshape((-1, 1)).as_in_context(context)

        output, decoded = net(*data[:-1])
        f1.update(data[len(data) - 1], output)

    return float(f1.get()[1])


if __name__ == '__main__':
    options = utils.parse_args()
    ctx = mx.gpu(options.gpu_index) if options.gpu_index >= 0 else mx.cpu()

    print('Receiving data')
    df = get_data_frame()
    train_df, val_df, test_df = np.split(df.sample(frac=1), [int(.8 * len(df)), int(.9 * len(df))])
    print('Train: {} records, Val: {} records, Test: {} records'.format(len(train_df),
                                                                        len(val_df),
                                                                        len(test_df)))

    train_dataloader = DataLoader(TelcoDataset(train_df), batch_size=options.batch_size,
                                  shuffle=True, last_batch='keep', num_workers=5, pin_memory=True)
    val_dataloader = DataLoader(TelcoDataset(val_df), batch_size=options.batch_size, shuffle=False,
                                last_batch='keep', num_workers=5, pin_memory=True)
    test_dataloader = DataLoader(TelcoDataset(test_df), batch_size=options.batch_size,
                                 shuffle=False, last_batch='keep', num_workers=5, pin_memory=True)

    net = TelcoModel()
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    net.hybridize(static_alloc=True, static_shape=True)

    print('Training started')
    train(net, train_dataloader, val_dataloader, options.epochs, ctx)
    net.load_parameters('best_model.params')
    test_f1 = evaluate(net, test_dataloader, ctx)
    print('Test F1: {:.3f}'.format(test_f1))
