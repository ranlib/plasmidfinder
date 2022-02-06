#!/usr/bin/env python
"""
Model class
"""
import os
import time
import configparser
import numpy
import tensorflow
from tensorflow import keras


class TimeHistory(keras.callbacks.Callback):
    """record time history of fit"""

    def __init__(self):
        super().__init__()
        self.times = []
        self.epoch_time_start = 0

    def on_train_begin(self, logs: dict = None):
        """initialize training"""
        self.times = []

    def on_epoch_begin(self, epoch: int, logs: dict = None):
        """initialize epoch"""
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch: int, logs: dict = None):
        """clean up epoch"""
        self.times.append(time.time() - self.epoch_time_start)


class Model:
    """model class"""

    def __init__(self, configuration: configparser.ConfigParser()):
        """constructor"""
        self.configuration = configuration
        self.subsequences1hot_trn = numpy.empty([], dtype=numpy.float32)
        self.subsequences1hot_val = numpy.empty([], dtype=numpy.float32)
        self.featuresXsubsequences_trn = numpy.empty([], dtype=numpy.float32)
        self.featuresXsubsequences_val = numpy.empty([], dtype=numpy.float32)
        self.labelsXsubsequences_trn = numpy.empty([], dtype=numpy.float32)
        self.labelsXsubsequences_val = numpy.empty([], dtype=numpy.float32)
        self.model = []

    def predict(self, list_of_subsequences: numpy.array, features: numpy.array) -> []:
        """
        prediction
        need to copy python lists into numpy arrays for keras.model.predict
        """
        # loop over all models and call prediction
        # Note: EagerTensor -> numpy array conversion
        predictions = []
        for _model in self.model:
            prediction = _model(
                [list_of_subsequences, features],
                self.configuration["PREDICTION"].getint("verbose"),
            )  # EagerTensor
            predictions.append(prediction.numpy())  # list of numpy.arrays

        return predictions

    def build(self):
        """
        Build model
        """
        shape_a = self.subsequences1hot_trn.shape
        shape_b = self.featuresXsubsequences_trn.shape

        input_a = keras.layers.Input(
            shape=(shape_a[1], shape_a[2]),
            name=f"sequences_{shape_a[1]}_x_{shape_a[2]}",
        )
        input_b = keras.layers.Input(shape=(shape_b[1],), name=f"features_{shape_b[1]}")

        # input_a is the 1-hot encoded sequences and it goes into an LSTM.
        lstm_dim = 40
        recdropout = self.configuration["TRAINING"].getfloat("recdropfrac")
        dropout = self.configuration["TRAINING"].getfloat("dropfrac")
        net = keras.layers.LSTM(
            lstm_dim,
            activation="tanh",
            recurrent_dropout=recdropout,
            dropout=dropout,
            name=f"A_{lstm_dim}",
            return_sequences=True,
        )(input_a)
        net = keras.layers.LSTM(
            lstm_dim,
            activation="tanh",
            recurrent_dropout=recdropout,
            dropout=dropout,
            name=f"B_{lstm_dim}",
        )(net)

        # input_b is the statistical features and it goes into a Dense (CNN) network.
        dens_dim = 10
        net2 = keras.layers.Dense(dens_dim, activation="tanh", name="G_4")(input_b)
        net2 = keras.layers.Dense(dens_dim, activation="tanh", name="H_4")(net2)

        # the two above get concatenated.
        net = keras.layers.concatenate([net, net2], name="seq_glob")

        net = keras.layers.Dense(
            dens_dim * 2, activation="relu", name=f"C_{dens_dim*2}"
        )(net)
        net = keras.layers.Dropout(dropout, name=f"fr_{dropout:.1f}")(net)
        net = keras.layers.Dense(dens_dim, activation="relu", name=f"D_{dens_dim}")(net)
        net = keras.layers.Dropout(dropout, name="fr_same")(net)
        outputs = keras.layers.Dense(1, activation="sigmoid", name="score")(net)

        self.model = keras.Model(
            inputs=[input_a, input_b], outputs=outputs, name="deeplasmid"
        )

        metrics = ["accuracy"]
        metrics.append(keras.metrics.SensitivityAtSpecificity(0.5))
        metrics.append(keras.metrics.TruePositives())
        metrics.append(keras.metrics.FalsePositives())
        metrics.append(
            keras.metrics.AUC(
                num_thresholds=200,
                curve="ROC",
                summation_method="interpolation",
                name=None,
                dtype=None,
                thresholds=None,
                multi_label=False,
                label_weights=None,
            )
        )
        self.model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=metrics
        )

    def train(self):
        """train the model"""
        callbacks_list = []
        callbacks_list.append(TimeHistory())

        if self.configuration["TRAINING"].getint("earlystoppatience") > 0:
            early_stop = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.configuration["TRAINING"].getint("earlystoppatience"),
                verbose=1,
                min_delta=1.0e-5,
                mode="auto",
            )
            callbacks_list.append(early_stop)

        if self.configuration["TRAINING"].getboolean("do_check_point"):
            check_point_file = os.path.join(
                self.configuration["LOGGING"]["output_directory"],
                self.configuration["TRAINING"]["training_output_directory"],
                self.configuration["TRAINING"]["check_point_file"],
            )
            check_point = keras.callbacks.ModelCheckpoint(
                check_point_file,
                monitor="val_loss",
                save_best_only=False,
                save_weights_only=True,
                verbose=1,
                save_freq=1,
            )
            callbacks_list.append(check_point)

        if self.configuration["TRAINING"].getboolean("do_reduce_learning"):
            reduce_learning = keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.3,
                patience=4,
                min_lr=0.0,
                verbose=1,
                min_delta=0.003,
            )
            callbacks_list.append(reduce_learning)

        if self.configuration["TRAINING"].getboolean("do_csv_logger"):
            csv_logger_file = os.path.join(
                self.configuration["LOGGING"]["output_directory"],
                self.configuration["TRAINING"]["training_output_directory"],
                self.configuration["TRAINING"]["csv_logger_file"],
            )
            csv_logger = keras.callbacks.CSVLogger(
                csv_logger_file, separator="\t", append=False
            )
            callbacks_list.append(csv_logger)

        if self.configuration["TRAINING"].getboolean("do_tensorboard"):
            tensorboard_log_directory = os.path.join(
                self.configuration["LOGGING"]["output_directory"],
                self.configuration["TRAINING"]["training_output_directory"],
                self.configuration["TRAINING"]["tensorboard_log_directory"],
            )
            tensor_board = keras.callbacks.TensorBoard(
                log_dir=tensorboard_log_directory,
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq=1,
            )
            callbacks_list.append(tensor_board)

        history = self.model.fit(
            [self.subsequences1hot_trn, self.featuresXsubsequences_trn],
            self.labelsXsubsequences_trn,
            validation_data=(
                [self.subsequences1hot_val, self.featuresXsubsequences_val],
                self.labelsXsubsequences_val,
            ),
            callbacks=callbacks_list,
            shuffle=True,
            batch_size=self.configuration["TRAINING"].getint("batch_size"),
            epochs=self.configuration["TRAINING"].getint("epochs"),
            steps_per_epoch=self.configuration["TRAINING"].getint("steps_per_epoch"),
            verbose=self.configuration["TRAINING"].getint("verbose"),
            workers=self.configuration["TRAINING"].getint("workers"),
            use_multiprocessing=self.configuration["TRAINING"].getboolean(
                "use_multiprocessing"
            ),
        )
        return history


class Model2(keras.Model):
    """model2 class"""

    def __init__(self, configuration: configparser.ConfigParser()):
        """constructor"""
        super().__init__()

        self.configuration = configuration

        dens_dim = 10
        lstm_dim = 40

        recdropout = self.configuration["TRAINING"].getfloat("recdropfrac")
        dropout = self.configuration["TRAINING"].getfloat("dropfrac")

        self.a40 = keras.layers.LSTM(
            lstm_dim,
            activation=tensorflow.nn.tanh,
            recurrent_dropout=recdropout,
            dropout=dropout,
            name=f"A_{lstm_dim}",
            return_sequences=True,
        )
        self.b40 = keras.layers.LSTM(
            lstm_dim,
            activation=tensorflow.nn.tanh,
            recurrent_dropout=recdropout,
            dropout=dropout,
            name=f"B_{lstm_dim}",
        )

        self.g4 = keras.layers.Dense(
            dens_dim, activation=tensorflow.nn.tanh, name="G_4"
        )
        self.h4 = keras.layers.Dense(
            dens_dim, activation=tensorflow.nn.tanh, name="H_4"
        )

        self.c100 = keras.layers.Dense(
            dens_dim * 2, activation=tensorflow.nn.relu, name=f"C_{dens_dim*2}"
        )
        self.fr = keras.layers.Dropout(dropout, name=f"fr_{dropout:.1f}")

        self.d10 = keras.layers.Dense(
            dens_dim, activation=tensorflow.nn.relu, name=f"D_{dens_dim}"
        )
        self.fr_same = keras.layers.Dropout(dropout, name="fr_same")

        self.outputs = keras.layers.Dense(1, activation="sigmoid", name="score")

    def call(self, inputs: list, training: bool = False):
        """run data through neural network"""
        x = self.b40(self.a40(inputs[0]))
        y = self.h4(self.g4(inputs[1]))
        z = keras.layers.Concatenate()([x, y])

        c = self.c100(z)
        if training:
            c = self.fr(c, training=training)

        d = self.d10(c)
        if training:
            d = self.fr_same(d, training=training)

        return self.outputs(d)
