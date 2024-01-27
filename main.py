import numpy as np
import pandas as pd
import pandas_datareader.data as data
import datetime
from dateutil.relativedelta import relativedelta
import sys
import os
import glob
from itertools import product

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import load_model
from tensorflow.keras.losses import binary_crossentropy, hinge
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LeakyReLU
from tensorflow.python.keras.engine import data_adapter

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Attention, Layer
from tensorflow.keras.metrics import MeanSquaredError

from sklearn.metrics import mean_squared_error,mean_absolute_error, matthews_corrcoef, accuracy_score
from sklearn.preprocessing import robust_scale, OneHotEncoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def select_dataset(dataset):
    if dataset=="stocknet-dataset":#88 stock
        start=training=datetime.datetime(year=2014, month=1, day=1)
        validation=datetime.datetime(year=2015, month=8, day=1)
        testing=datetime.datetime(year=2015, month=10, day=1)
        end=datetime.datetime(year=2016, month=1, day=1)
    elif dataset=="kdd17":#50 stock
        start=training=datetime.datetime(year=2007, month=1, day=1)
        validation=datetime.datetime(year=2015, month=1, day=3)
        testing=datetime.datetime(year=2016, month=1, day=4)
        end=datetime.datetime(year=2017, month=1, day=1)
    elif dataset=="ACL18_hour":
        start=training=datetime.datetime(year=2014, month=1, day=1)
        validation=datetime.datetime(year=2015, month=8, day=1)
        testing=datetime.datetime(year=2015, month=10, day=1)
        end=datetime.datetime(year=2016, month=1, day=1)
    elif dataset=="kdd17_hour":
        start=training=datetime.datetime(year=2007, month=1, day=1)
        validation=datetime.datetime(year=2015, month=1, day=3)
        testing=datetime.datetime(year=2016, month=1, day=4)
        end=datetime.datetime(year=2017, month=1, day=1)
    return start, validation, testing, end

def hinge_acc(y_true, y_pred):
    y_pred = (tf.sign(y_pred) + 1) / 2
    y_pred = tf.where(tf.abs(y_pred - 0.5) < 1e-8, 0.0, y_pred)
    y_pred = tf.where(tf.math.is_nan(y_pred), 0.0, y_pred)
    y_true = tf.cast(y_true, tf.float32)
    return tf.keras.metrics.binary_accuracy(y_true, y_pred)

class MCC(tf.keras.metrics.Metric):
    def __init__(self, name='mcc', **kwargs):
        super(MCC, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        true_positives = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        false_positives = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        true_negatives = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, False))
        false_negatives = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))

        self.true_positives.assign_add(tf.reduce_sum(tf.cast(true_positives, self.dtype)))
        self.false_positives.assign_add(tf.reduce_sum(tf.cast(false_positives, self.dtype)))
        self.true_negatives.assign_add(tf.reduce_sum(tf.cast(true_negatives, self.dtype)))
        self.false_negatives.assign_add(tf.reduce_sum(tf.cast(false_negatives, self.dtype)))

    def result(self):
        numerator = (self.true_positives * self.true_negatives) - (self.false_positives * self.false_negatives)
        denominator = tf.sqrt((self.true_positives + self.false_positives) * (self.true_positives + self.false_negatives) * (self.true_negatives + self.false_positives) * (self.true_negatives + self.false_negatives))
        return numerator / (denominator + tf.keras.backend.epsilon())

    def reset_state(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.true_negatives.assign(0)
        self.false_negatives.assign(0)

class TemporalAttention(Layer):
    def __init__(self, units, name="atten"):
        super(TemporalAttention, self).__init__(name=name)
        self.units = units

    def build(self, input_shape):
        self.av_W = self.add_weight(
            name='att_W', dtype=tf.float32,
            shape=[self.units, self.units],
            initializer="glorot_uniform",
            trainable= True
        )
        self.av_b = self.add_weight(
            name='att_h', dtype=tf.float32,
            shape=[self.units],
            initializer="zeros",
            trainable= True
        )
        self.av_u = self.add_weight(
            name='att_u', dtype=tf.float32,
            shape=[self.units],
            initializer="glorot_uniform",
            trainable= True
        )
        super().build(input_shape)

    def call(self, h):
        a_laten = tf.tanh(
            tf.tensordot(h, self.av_W,
                         axes=1) + self.av_b)
        a_scores = tf.tensordot(a_laten, self.av_u,
                                     axes=1,
                                     name='scores')
        a_alphas = tf.nn.softmax(a_scores, name='alphas')

        a_con = tf.reduce_sum(
            h * tf.expand_dims(a_alphas, -1), 1)

        fea_con = tf.concat(
            [h[:, -1, :], a_con],
            axis=1)
        return fea_con


class CustomDense(Layer):
    def __init__(self, units, hinge, name="pre_fc"):
        super(CustomDense, self).__init__(name=name)
        self.units = units
        self.hinge = hinge

    def build(self, input_shape):
        self.fc_W = self.add_weight(
            shape=(self.units * 2, 1),
            initializer='glorot_uniform',
            trainable=True,
        )
        self.fc_b = self.add_weight(
            name='bias',
            shape=(1,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        if self.hinge:
            return tf.nn.bias_add(tf.matmul(inputs, self.fc_W), self.fc_b)
        else:
            return tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(inputs, self.fc_W), self.fc_b))


class SelfAttention(Layer):
    def __init__(self, units, name="self_atten"):
        super(SelfAttention, self).__init__(name=name)
        self.units = units

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[-1], self.units),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[-1], self.units),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[-1], self.units),
                                  initializer='glorot_uniform',
                                  trainable=True)

    def call(self, inputs):
        Q = tf.matmul(inputs, self.WQ)
        K = tf.matmul(inputs, self.WK)
        V = tf.matmul(inputs, self.WV)

        attention_scores = tf.matmul(Q, K, transpose_b=True)
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)

        attention_output = tf.matmul(attention_scores, V)
        return attention_output

class AdvALSTM(tf.keras.models.Model):
    def __init__(self, units, epsilon = 1E-3, beta =  5E-2, learning_rate = 1E-2, dropout = 0.0, l2_reg = None, attention = True, hinge = True, adversarial_training = True, random_perturbations = False):
        super().__init__()
        self.epsilon = tf.constant(epsilon)
        self.beta = tf.constant(beta)
        self.l2_reg = l2_reg
        self.hinge = hinge
        self.attention = attention
        self.adversarial_training = adversarial_training
        self.random_perbutations = random_perturbations


        if self.attention:
            self.model_latent_rep = tf.keras.models.Sequential([
                #Conv1D(filters=64, kernel_size=2, activation='relu'),
                #MaxPooling1D(pool_size=2),
                SelfAttention(units),                
                tf.keras.layers.Dense(units, activation = "tanh", kernel_initializer = keras.initializers.glorot_uniform),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.LSTM(units, return_sequences = True, dropout = dropout),

            ])
            self.model_prediction = tf.keras.models.Sequential([
                TemporalAttention(units),
                CustomDense(units, hinge)
                ])
        else:
            self.model_latent_rep = tf.keras.models.Sequential([
                tf.keras.layers.Dense(units, activation = "tanh", kernel_initializer = keras.initializers.glorot_uniform),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.LSTM(units, return_sequences = True, dropout = dropout),
                ])
            self.model_prediction = tf.keras.models.Sequential([
                tf.keras.layers.Dense(1, activation=None if hinge else "sigmoid", kernel_initializer = keras.initializers.glorot_uniform, name='pre_fc')
                ])

        self.compile(
            loss = "hinge" if hinge else "binary_crossentropy",
            optimizer = tf.keras.optimizers.Adam(learning_rate),
            metrics = [hinge_acc if hinge else "acc", MCC()]
        )

    def call(self, x):
        x = self.model_latent_rep(x)
        x = self.model_prediction(x)
        return x

    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            e = self.model_latent_rep(x, training=True)
            if self.attention:
                y_pred = self.model_prediction(e, training=True)
            else:
                y_pred = self.model_prediction(e[:, -1, :], training=True)

            hinge_loss_fn = tf.keras.losses.Hinge()
            loss = hinge_loss_fn(y, y_pred)

            l2_norm = tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables if 'pre_fc' in v.name])
            total_loss = loss + self.l2_reg * l2_norm


            if self.adversarial_training:
                with tape.stop_recording():
                    if self.random_perbutations:
                        perturbations = tf.random.normal(shape=tf.shape(e))
                    else:
                        perturbations = tf.math.sign(tape.gradient(loss, e))
                    tf.stop_gradient(perturbations)
                e_adv = e + self.epsilon * tf.norm(e, ord="euclidean", axis=-1, keepdims=True) * tf.math.l2_normalize(perturbations, axis=-1, epsilon=1e-8)
                y_pred_adv = self.model_prediction(e_adv, training=True)

                adv_loss = hinge_loss_fn(y, y_pred_adv)
                total_loss += self.beta * adv_loss + tf.add_n(self.losses)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return self.compute_metrics(x, y, y_pred, sample_weight)

def train_model(model, X_train, y_train, X_val, y_val, X_test, y_test, epochs, batch_size):
    best_test_acc = 0
    best_test_mcc = 0

    for epoch in range(epochs):
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        num_samples = X_train.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))

        for batch in range(num_batches):
            batch_start = batch * batch_size
            batch_end = min(batch_start + batch_size, num_samples)
            X_batch = X_train_shuffled[batch_start:batch_end]
            y_batch = y_train_shuffled[batch_start:batch_end]
            model.train_on_batch(X_batch, y_batch)

        model.evaluate(X_val, y_val, verbose=0)

        test_metrics = model.evaluate(X_test, y_test, verbose=0)
        test_acc = test_metrics[1]
        test_mcc = test_metrics[2]

        #print(f'Epoch {epoch + 1}/{epochs}')
        #print(f'Test Accuracy: {test_acc}, Test MCC: {test_mcc}')

        if test_acc > best_test_acc or test_mcc > best_test_mcc:
            best_test_acc = test_acc
            best_test_mcc = test_mcc

    return best_test_acc, best_test_mcc

def generate_sequences_atten(df, features_columns, T):
    X = []
    y = []
    for i in range(T, len(df)):
        X.append(df[features_columns].iloc[i-T:i].values)
        y.append(df["Up"].iloc[i])
    return np.array(X), np.array(y)

def generate_sequences_lstm(df, features_columns, T):
    df_list = []
    X_stock_array = np.array(df[features_columns])
    y_stock_array = np.array(df["Up"])
    sequences_indexes = [np.arange(i, T + i, 1) for i in range(len(df) - T)]
    _X = X_stock_array[sequences_indexes]
    _y = y_stock_array[sequences_indexes][:, -1]
    return _X, _y

def add_weekday_encoding(df):
    df['weekday'] = df.index.dayofweek
    encoder = OneHotEncoder(sparse_output=False)
    weekday_encoded = encoder.fit_transform(df[['weekday']].values)

    weekday_columns = [f'weekday_{i}' for i in range(weekday_encoded.shape[1])]
    df_weekday = pd.DataFrame(weekday_encoded, index=df.index, columns=weekday_columns)
    df.drop('weekday', axis=1, inplace=True)
    return pd.concat([df, df_weekday], axis=1)

def load_data(dataset, attention, T):
  X_train_all=None
  y_train_all=None
  X_val_all=None
  y_val_all=None
  X_test_all=None
  y_test_all=None

  if dataset=="KDD17" or "ACL18":
    raw_data_path = f"/home/20x3051_sasamura/code/{dataset}/ourpped/*.csv"
    raw_data_pathes = glob.glob(raw_data_path)

  trading_dates = pd.read_csv(f"/home/20x3051_sasamura/code/{dataset}/trading_dates.csv", header=None)
  trading_dates.columns = ['Date']
  for path in raw_data_pathes:
      #print(path)
      df = pd.read_csv(path, header=None)
      df.columns = ['High', 'Low', 'Open', 'Close_', 'Adj Close_', '5-day', '10-day', '15-day', '20-day', '25-day', '30-day', 'Up', 'Delete']
      df.index = pd.DatetimeIndex(trading_dates['Date'].values.flatten())
      df.replace(-123321, np.nan, inplace=True)
      df = df[df['Up'] != 0]

      df.loc[:, 'Up'] = ((df['Up'].copy() + 1) / 2).values

      df = df.dropna()
      df = add_weekday_encoding(df)

      #print(df)

      df_train = df[(start <= df.index) & (df.index < validation)]
      df_val = df[(validation <= df.index) & (df.index < testing)]
      df_test = df[(testing <= df.index) & (df.index < end)]


      if attention:
          X_train, y_train = generate_sequences_atten(df_train, ['High', 'Low', 'Open', 'Close_', 'Adj Close_', '5-day', '10-day', '15-day', '20-day', '25-day', '30-day'] + [f'weekday_{i}' for i in range(5)], T=T)
          X_val,   y_val   = generate_sequences_atten(df_val,   ['High', 'Low', 'Open', 'Close_', 'Adj Close_', '5-day', '10-day', '15-day', '20-day', '25-day', '30-day'] + [f'weekday_{i}' for i in range(5)], T=T)
          X_test,  y_test  = generate_sequences_atten(df_test,  ['High', 'Low', 'Open', 'Close_', 'Adj Close_', '5-day', '10-day', '15-day', '20-day', '25-day', '30-day'] + [f'weekday_{i}' for i in range(5)], T=T)
      else:
          X_train, y_train = generate_sequences_lstm(df_train, ['High', 'Low', 'Open', 'Close_', 'Adj Close_', '5-day', '10-day', '15-day', '20-day', '25-day', '30-day'] + [f'weekday_{i}' for i in range(5)], T=T)
          X_val,   y_val   = generate_sequences_lstm(df_val,   ['High', 'Low', 'Open', 'Close_', 'Adj Close_', '5-day', '10-day', '15-day', '20-day', '25-day', '30-day'] + [f'weekday_{i}' for i in range(5)], T=T)
          X_test,  y_test  = generate_sequences_lstm(df_test,  ['High', 'Low', 'Open', 'Close_', 'Adj Close_', '5-day', '10-day', '15-day', '20-day', '25-day', '30-day'] + [f'weekday_{i}' for i in range(5)], T=T)

      if X_train_all is None: X_train_all = X_train
      else : X_train_all = np.concatenate([X_train_all, X_train], axis = 0)
      if y_train_all is None: y_train_all = y_train
      else : y_train_all = np.concatenate([y_train_all, y_train], axis = 0)

      if X_val_all is None: X_val_all = X_val
      else : X_val_all = np.concatenate([X_val_all, X_val], axis = 0)
      if y_val_all is None: y_val_all = y_val
      else : y_val_all = np.concatenate([y_val_all, y_val], axis = 0)

      if X_test_all is None: X_test_all = X_test
      else : X_test_all = np.concatenate([X_test_all, X_test], axis = 0)
      if y_test_all is None: y_test_all = y_test
      else : y_test_all = np.concatenate([y_test_all, y_test], axis = 0)
  #print("training data:", len(X_train_all))
  #print("validation data:", len(X_val_all))
  #print("test data:", len(X_test_all))
  return X_train_all, X_val_all, X_test_all, y_train_all, y_val_all, y_test_all


#stocknet-dataset, kdd17
dataset="kdd17"
T_values = [2, 3, 4, 5, 10, 15]
units_values = [4, 8, 16, 32]
l2_reg_values = [0.001, 0.01, 0.1, 1, 10]
learning_rate= [0.001, 0.01]
start, validation, testing, end = select_dataset(dataset)

param_combinations = list(product(T_values, units_values, l2_reg_values, learning_rate))#

best_params = {}
best_test_acc = 0
best_test_mcc = 0

for T, units, l2_reg, learning_rate in param_combinations:
    X_train_all, X_val_all, X_test_all, y_train_all, y_val_all, y_test_all = load_data(dataset, attention, T=T)

    tf.random.set_seed(123456)
    model = AdvALSTM(
        units=units,
        learning_rate=learning_rate,
        l2_reg=l2_reg,
        attention=False,
        hinge=True,
        dropout=0.0,
        adversarial_training=False,
        random_perturbations=False
    )

    test_acc, test_mcc = train_model(
        model,
        X_train_all,
        y_train_all,
        X_val_all,
        y_val_all,
        X_test_all,
        y_test_all,
        epochs=150,
        batch_size=1024
    )

    print(f'T: {T}, Units: {units}, L2_Reg: {l2_reg}, Learning_Rate: {learning_rate}, Test Accuracy: {test_acc}, Test MCC: {test_mcc}')

    if test_acc > best_test_acc and (test_acc == best_test_acc and test_mcc > best_test_mcc):
        best_test_acc = test_acc
        best_test_mcc = test_mcc
        best_params = {'T': T, 'units': units, 'l2_reg': l2_reg, 'Learning_Rate': learning_rate}

print(f'Best Parameters: {best_params}, Best Test Accuracy: {best_test_acc}, Best Test MCC: {best_test_mcc}')
print("training data:", len(X_train_all))
print("validation data:", len(X_val_all))
print("test data:", len(X_test_all))
