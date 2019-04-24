from mxnet import gluon
from mxnet.gluon import HybridBlock
from mxnet.gluon.nn import Dense


class TelcoModel(HybridBlock):
    def __init__(self):
        super().__init__()

        with self.name_scope():
            self.encoder = gluon.nn.HybridSequential(prefix="")
            self.decoder = gluon.nn.HybridSequential(prefix="")

            with self.encoder.name_scope():
                self.encoder.add(gluon.nn.BatchNorm(epsilon=1e-05, momentum=0.1,
                                                    use_global_stats=True))
                self.encoder.add(Dense(128, activation='relu'))
                self.encoder.add(gluon.nn.BatchNorm(epsilon=1e-05, momentum=0.1,
                                                    use_global_stats=True))

                self.encoder.add(Dense(64, activation='relu'))
                self.encoder.add(gluon.nn.BatchNorm(epsilon=1e-05, momentum=0.1,
                                                    use_global_stats=True))

                self.encoder.add(Dense(32, activation='relu'))
                self.encoder.add(gluon.nn.BatchNorm(epsilon=1e-05, momentum=0.1,
                                                    use_global_stats=True))

                self.encoder.add(Dense(16, activation='relu'))

            with self.decoder.name_scope():
                self.decoder.add(gluon.nn.BatchNorm(epsilon=1e-05, momentum=0.1,
                                                    use_global_stats=True))

                self.decoder.add(Dense(32, activation='relu'))
                self.decoder.add(gluon.nn.BatchNorm(epsilon=1e-05, momentum=0.1,
                                                    use_global_stats=True))

                self.decoder.add(Dense(64, activation='relu'))
                self.decoder.add(gluon.nn.BatchNorm(epsilon=1e-05, momentum=0.1,
                                                    use_global_stats=True))

                self.decoder.add(Dense(128, activation='relu'))
                self.decoder.add(gluon.nn.BatchNorm(epsilon=1e-05, momentum=0.1,
                                                    use_global_stats=True))

                self.decoder.add(Dense(26))

            self.output = Dense(2)

    def hybrid_forward(self, F, *args):
        data = F.concat(*args, dim=1)
        encoded_data = self.encoder(data)
        decoded_data = self.decoder(encoded_data)
        out = self.output(encoded_data)
        return out, decoded_data
