import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dropout', type=float, default=0.0)

    # for full list of pretrained language models see:
    # http://gluon-nlp.mxnet.io/model_zoo/language_model/index.html
    # Try awd_lstm_lm_1150 for better results (but takes way longer to train)
    parser.add_argument('--lm_name', type=str, default='standard_lstm_lm_200')

    parser.add_argument('--lm_dataset', type=str, default='wikitext-2')
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--bucket_num', type=int, default=10)
    parser.add_argument('--bucket_ratio', type=int, default=0.2)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--gpu_index', type=int, default=0)

    return parser.parse_args()