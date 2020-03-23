# Copyright 2020 Cognoscitive Automata Incorporated
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
os.environ['PYTHONHASHSEED'] = str(0)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import sys
import random
random.seed(0)
import time
from urllib.request import urlopen
from datetime import datetime, timedelta
import gc
import numpy as np
np.random.seed(0)
from PIL import Image
import tensorflow as tf
tf.random.set_seed(0)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, add, concatenate, Conv2D, MaxPooling2D, Flatten, Reshape, Conv2DTranspose, UpSampling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN
import logging
from losses import exp_or_log, linex, smooth_l1

logging.basicConfig(level=logging.INFO)

HIDDEN_ACT = 'tanh'
GATE_ACT = 'sigmoid'
OUTPUT_ACT = 'linear'
DATA_PATH = 'data/earthquake'
TODAY = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
START_DATE = datetime(1973, 1, 1)
END_DATE = datetime(2018, 1, 1)
TOTAL_TIME = (END_DATE - START_DATE).total_seconds()
TEST_START_DATE = datetime(2018, 1, 1)
TEST_END_DATE = datetime(2019, 1, 1)
TEST_TOTAL_TIME = (TEST_END_DATE - TEST_START_DATE).total_seconds()
UPDATE_START_DATE = datetime(2019, 1, 1)
UPDATE_END_DATE = TODAY - timedelta(days=1)
UPDATE_TOTAL_TIME = (UPDATE_END_DATE - UPDATE_START_DATE).total_seconds()
SEED_TOTAL_TIME = 24*60*60
SECONDS_PER_TIMESTEP = 24*60*60
LATITUDES = 90
LONGITUDES = 180
FEATURES = 1
EMBEDDING_DIMS = 256
HIDDEN_NODES = 512
SEQ_LEN = 1
DATA_STEP = 1
STATEFUL = True
DROPOUT_PROB = 0.0
TIMESKIP_PROB = 0.01
BATCH_SIZE = 1
EPOCHS = 1
MODEL_NAMES = ['high-recall', 'high-precision']
MODEL_FILE = 'model/model_v3'
STATE_FILE = 'model/state_v3'
TEST_FILE = 'test/stats_v3'
LOSS = exp_or_log
MODEL_FILES = ['model/model_exporlog_2020', 'model/model_smoothl1_2020']
STATE_FILES = ['model/state_exporlog_2020', 'model/state_smoothl1_2020']
TEST_FILES = ['test/stats_exporlog_2020', 'test/stats_smoothl1_2020']
LOSSES = [exp_or_log, smooth_l1]
TEST_MAGNITUDE_CUTOFF = -1.0
TEST_MAGNITUDE_RANGE = 1.0
TEST_COORDINATE_RANGE = 1.0
THRESHOLD_MAGNITUDE = -1.0
PREDICTION_MODE = 'regression'
#PREDICTION_MODE = 'classification'
CONVOLUTION = False
POOLING = False
ONEHOT_LENGTH = 10
POSITIVE_PROBABILITY_THRESHOLD = 0.5
PREDICTION_MAGNITUDE_CUTOFF = 0.0
STATEFUL_LAYER_NAMES = ['lstm1', 'lstm2', 'lstm3', 'lstm4', 'lstm5']

CUSTOM_OBJECTS = {'exp_or_log': exp_or_log, 'linex': linex, 'smooth_l1': smooth_l1}

def estimate_interval():
    max_interval = 0
    last_time = START_DATE
    for year in range(START_DATE.year, END_DATE.year):
        for month in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
            with open(DATA_PATH + str(year) + '-' + month + '.csv', 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    earthquake = line.split(',')
                    timestep = (datetime.strptime(earthquake[0], '%Y-%m-%dT%H:%M:%S.%fZ') - last_time).total_seconds()
                    last_time = datetime.strptime(earthquake[0], '%Y-%m-%dT%H:%M:%S.%fZ')
                    if timestep > max_interval:
                        max_interval = timestep
    print(str(max_interval)) #49097.9

def estimate_timesteps():
    max_timesteps = 0
    timesteps = 0
    last_time = START_DATE
    for year in range(START_DATE.year, END_DATE.year):
        for month in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
            with open(DATA_PATH + str(year) + '-' + month + '.csv', 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    earthquake = line.split(',')
                    this_time = datetime.strptime(earthquake[0], '%Y-%m-%dT%H:%M:%S.%fZ')
                    if this_time.day == last_time.day:
                        timesteps += 1
                    else:
                        timesteps = 0
                    last_time = this_time
                    if timesteps > max_timesteps:
                        max_timesteps = timesteps
    print(str(max_timesteps))

def pad_number(number):
    if number < 10:
        return '0' + str(number)
    return str(number)

def download_month(year, month):
    if month < 12:
        end_year = year
        end_month = month + 1
    elif month == 12:
        end_year = year + 1
        end_month = 1
    data = urlopen('https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime='+str(year)+'-'+pad_number(month)+'-01&endtime='+str(end_year)+'-'+pad_number(end_month)+'-01')
    with open(DATA_PATH + str(year) + '-' + pad_number(month) + '.csv', 'w') as f:
        f.write(data.read().decode('utf-8'))

def download_day(year, month, day):
    if month == 12 and day == 31:
        end_year = year + 1
        end_month = 1
        end_day = 1
    elif month in [1, 3, 5, 7, 8, 10] and day == 31:
        end_year = year
        end_month = month + 1
        end_day = 1
    elif month in [4, 6, 9, 11] and day == 30:
        end_year = year
        end_month = month + 1
        end_day = 1
    elif year % 4 == 0 and month == 2 and day == 29:
        end_year = year
        end_month = month + 1
        end_day = 1
    elif year % 4 != 0 and month == 2 and day == 28:
        end_year = year
        end_month = month + 1
        end_day = 1
    else:
        end_year = year
        end_month = month
        end_day = day + 1
    data = urlopen('https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime='+str(year)+'-'+pad_number(month)+'-'+pad_number(day)+'&endtime='+str(end_year)+'-'+pad_number(end_month)+'-'+pad_number(end_day))
    with open(DATA_PATH + str(year) + '-' + pad_number(month) + '-' + pad_number(day) + '.csv', 'w') as f:
        f.write(data.read().decode('utf-8'))

def download_dataset():
    year = START_DATE.year
    month = 1
    while year < END_DATE.year:
        try:
            download_month(year, month)
            time.sleep(random.randint(5, 10))
            if month < 12:
                month = month + 1
            elif month == 12:
                year += 1
                month = 1
        except:
            print(sys.exc_info())
            time.sleep(random.randint(30, 60))

def download_update():
    year = TEST_START_DATE.year
    month = TEST_START_DATE.month
    day = TEST_START_DATE.day
    while year < TODAY.year or month < TODAY.month or day < TODAY.day:
        try:
            download_day(year, month, day)
            time.sleep(random.randint(3, 5))
            if month == 12 and day == 31:
                year += 1
                month = 1
                day = 1
            elif month in [1, 3, 5, 7, 8, 10] and day == 31:
                month += 1
                day = 1
            elif month in [4, 6, 9, 11] and day == 30:
                month += 1
                day = 1
            elif year % 4 == 0 and month == 2 and day == 29:
                month += 1
                day = 1
            elif year % 4 != 0 and month == 2 and day == 28:
                month += 1
                day = 1
            else:
                day += 1
        except:
            print(sys.exc_info())
            time.sleep(random.randint(10, 30))

def convert_magnitude(mag, mag_type):
    if len(mag_type) > 1:
        if mag_type[1] == 'w':
            return mag
        elif mag_type[1] == 's':
            if mag <= 5.5:
                return 0.5716 * mag + 2.4980
            else:
                return 0.8126 * mag + 1.1723
        elif mag_type[1] == 'b':
            return 1.0319 * mag + 0.0223
        elif mag_type[1] == 'd':
            return 0.7947 * mag + 1.3420
        elif mag_type[1] == 'l':
            return 0.8095 * mag + 1.3003
        else:
            return mag
    else:
        return mag

def magnitude2energy(magnitude):
    return 10 ** (magnitude * 1.5 + 9.1) / 20000

def energy2magnitude(energy):
    return (np.log10(energy * 20000) - 9.1) / 1.5

def initialize_energies(zeros):
    return zeros + magnitude2energy(THRESHOLD_MAGNITUDE)

def scale(magnitude):
    return (np.array(magnitude) + 1) / 11

def descale(magnitude):
    return (np.array(magnitude) * 11) - 1

def generate_dataset():
    data = np.zeros((int(TOTAL_TIME / SECONDS_PER_TIMESTEP), LATITUDES+1, LONGITUDES+1), dtype=np.float32)
    data = initialize_energies(data)
    for year in range(START_DATE.year, END_DATE.year):
        for month in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
            with open(DATA_PATH + str(year) + '-' + month + '.csv', 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    earthquake = line.split(',')
                    timestep = int((TOTAL_TIME - (END_DATE - datetime.strptime(earthquake[0], '%Y-%m-%dT%H:%M:%S.%fZ')).total_seconds()) / SECONDS_PER_TIMESTEP)
                    if earthquake[1] != '' and earthquake[2] != '' and earthquake[4] != '':
                        latitude = int(round((float(earthquake[1]) + LATITUDES) / 2))
                        longitude = int(round((float(earthquake[2]) + LONGITUDES) / 2))
                        magnitude = convert_magnitude(float(earthquake[4]), earthquake[5])
                        energy = magnitude2energy(magnitude)
                        data[timestep, latitude, longitude] += energy
    data = data.reshape((int(TOTAL_TIME / SECONDS_PER_TIMESTEP), (LATITUDES+1) * (LONGITUDES+1)))
    data = energy2magnitude(data)
    if OUTPUT_ACT == 'sigmoid':
        data = scale(data)
    np.save(DATA_PATH + '_data.npy', data)

def load_dataset():
    data = np.load(DATA_PATH + '_data.npy')
    if CONVOLUTION:
        data = data.reshape((int(TOTAL_TIME / SECONDS_PER_TIMESTEP), (LATITUDES+1), (LONGITUDES+1), 1))
    x = data
    y = data
    return x, y

def generate_testset():
    data = np.zeros((int(TEST_TOTAL_TIME / SECONDS_PER_TIMESTEP), LATITUDES+1, LONGITUDES+1), dtype=np.float32)
    data = initialize_energies(data)
    year = TEST_START_DATE.year
    month = TEST_START_DATE.month
    day = TEST_START_DATE.day
    while year < TEST_END_DATE.year or month < TEST_END_DATE.month or day < TEST_END_DATE.day:
        with open(DATA_PATH + str(year) + '-' + pad_number(month) + '-' + pad_number(day) + '.csv', 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                earthquake = line.split(',')
                timestep = int((TEST_TOTAL_TIME - (TEST_END_DATE - datetime.strptime(earthquake[0], '%Y-%m-%dT%H:%M:%S.%fZ')).total_seconds()) / SECONDS_PER_TIMESTEP)
                if earthquake[1] != '' and earthquake[2] != '' and earthquake[4] != '':
                    latitude = int(round((float(earthquake[1]) + LATITUDES) / 2))
                    longitude = int(round((float(earthquake[2]) + LONGITUDES) / 2))
                    magnitude = convert_magnitude(float(earthquake[4]), earthquake[5])
                    energy = magnitude2energy(magnitude)
                    data[timestep, latitude, longitude] += energy
        if month == 12 and day == 31:
            year += 1
            month = 1
            day = 1
        elif month in [1, 3, 5, 7, 8, 10] and day == 31:
            month += 1
            day = 1
        elif month in [4, 6, 9, 11] and day == 30:
            month += 1
            day = 1
        elif year % 4 == 0 and month == 2 and day == 29:
            month += 1
            day = 1
        elif year % 4 != 0 and month == 2 and day == 28:
            month += 1
            day = 1
        else:
            day += 1
    data = data.reshape((int(TEST_TOTAL_TIME / SECONDS_PER_TIMESTEP), (LATITUDES+1) * (LONGITUDES+1)))
    data = energy2magnitude(data)
    if OUTPUT_ACT == 'sigmoid':
        data = scale(data)
    np.save(DATA_PATH + '_test.npy', data)

def load_testset():
    data = np.load(DATA_PATH + '_test.npy')
    if CONVOLUTION:
        data = data.reshape((int(TEST_TOTAL_TIME / SECONDS_PER_TIMESTEP), (LATITUDES+1), (LONGITUDES+1), 1))
    x = []
    y = []
    for i in range(len(data) - SEQ_LEN):
        x.append(data[i:i+SEQ_LEN])
        y.append(data[i+SEQ_LEN:i+SEQ_LEN+1])
    return x, y

def generate_update():
    data = np.zeros((int(UPDATE_TOTAL_TIME / SECONDS_PER_TIMESTEP), LATITUDES+1, LONGITUDES+1), dtype=np.float32)
    data = initialize_energies(data)
    year = UPDATE_START_DATE.year
    month = UPDATE_START_DATE.month
    day = UPDATE_START_DATE.day
    while year < UPDATE_END_DATE.year or month < UPDATE_END_DATE.month or day < UPDATE_END_DATE.day:
        with open(DATA_PATH + str(year) + '-' + pad_number(month) + '-' + pad_number(day) + '.csv', 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                earthquake = line.split(',')
                timestep = int((UPDATE_TOTAL_TIME - (UPDATE_END_DATE - datetime.strptime(earthquake[0], '%Y-%m-%dT%H:%M:%S.%fZ')).total_seconds()) / SECONDS_PER_TIMESTEP)
                if earthquake[1] != '' and earthquake[2] != '' and earthquake[4] != '':
                    latitude = int(round((float(earthquake[1]) + LATITUDES) / 2))
                    longitude = int(round((float(earthquake[2]) + LONGITUDES) / 2))
                    magnitude = convert_magnitude(float(earthquake[4]), earthquake[5])
                    energy = magnitude2energy(magnitude)
                    data[timestep, latitude, longitude] += energy
        if month == 12 and day == 31:
            year += 1
            month = 1
            day = 1
        elif month in [1, 3, 5, 7, 8, 10] and day == 31:
            month += 1
            day = 1
        elif month in [4, 6, 9, 11] and day == 30:
            month += 1
            day = 1
        elif year % 4 == 0 and month == 2 and day == 29:
            month += 1
            day = 1
        elif year % 4 != 0 and month == 2 and day == 28:
            month += 1
            day = 1
        else:
            day += 1
    data = data.reshape((int(UPDATE_TOTAL_TIME / SECONDS_PER_TIMESTEP), (LATITUDES+1) * (LONGITUDES+1)))
    data = energy2magnitude(data)
    if OUTPUT_ACT == 'sigmoid':
        data = scale(data)
    np.save(DATA_PATH + '_update.npy', data)

def load_update():
    data = np.load(DATA_PATH + '_update.npy')
    if CONVOLUTION:
        data = data.reshape((int(TEST_TOTAL_TIME / SECONDS_PER_TIMESTEP), (LATITUDES+1), (LONGITUDES+1), 1))
    x = []
    y = []
    for i in range(len(data) - SEQ_LEN):
        x.append(data[i:i+SEQ_LEN])
        y.append(data[i+SEQ_LEN:i+SEQ_LEN+1])
    return x, y

def generate_seedset(today=TODAY):
    data = np.zeros((int(SEED_TOTAL_TIME / SECONDS_PER_TIMESTEP), LATITUDES+1, LONGITUDES+1), dtype=np.float32)
    data = initialize_energies(data)
    start = today - timedelta(days=SEQ_LEN)
    year = start.year
    month = start.month
    day = start.day
    while year < today.year or month < today.month or day < today.day:
        with open(DATA_PATH + str(year) + '-' + pad_number(month) + '-' + pad_number(day) + '.csv', 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                earthquake = line.split(',')
                timestep = int((SEED_TOTAL_TIME - (today - datetime.strptime(earthquake[0], '%Y-%m-%dT%H:%M:%S.%fZ')).total_seconds()) / SECONDS_PER_TIMESTEP)
                if earthquake[1] != '' and earthquake[2] != '' and earthquake[4] != '':
                    latitude = int(round((float(earthquake[1]) + LATITUDES) / 2))
                    longitude = int(round((float(earthquake[2]) + LONGITUDES) / 2))
                    magnitude = convert_magnitude(float(earthquake[4]), earthquake[5])
                    energy = magnitude2energy(magnitude)
                    data[timestep, latitude, longitude] += energy
        if month == 12 and day == 31:
            year += 1
            month = 1
            day = 1
        elif month in [1, 3, 5, 7, 8, 10] and day == 31:
            month += 1
            day = 1
        elif month in [4, 6, 9, 11] and day == 30:
            month += 1
            day = 1
        elif year % 4 == 0 and month == 2 and day == 29:
            month += 1
            day = 1
        elif year % 4 != 0 and month == 2 and day == 28:
            month += 1
            day = 1
        else:
            day += 1
    data = data.reshape((SEQ_LEN, (LATITUDES+1) * (LONGITUDES+1)))
    data = energy2magnitude(data)
    if OUTPUT_ACT == 'sigmoid':
        data = scale(data)
    np.save(DATA_PATH + '_seed.npy', data)

def load_seedset():
    data = np.load(DATA_PATH + '_seed.npy')
    return data

def data_generator(x, y, step=1):
    if PREDICTION_MODE == 'classification':
        x_int = np.clip(np.int8(x), 0, ONEHOT_LENGTH - 1)
        y_int = np.clip(np.int8(y), 0, ONEHOT_LENGTH - 1)
        identity = np.eye(ONEHOT_LENGTH, dtype=np.bool_)
    while(True):
        count = 0
        indices = range(0, len(x) - (SEQ_LEN - 1) - step, DATA_STEP)
        if not STATEFUL:
            random.shuffle(indices)
        for i in indices:
            count += 1
            if count == 1:
                x_batch = []
                y_batch = []
            if PREDICTION_MODE == 'regression':
                x_batch.append(x[i:i+SEQ_LEN])
                y_batch.append(y[i+step:i+step+SEQ_LEN]) # Sequence-to-Sequence
            elif PREDICTION_MODE == 'classification':
                x_batch.append(identity[x_int[i:i+SEQ_LEN]])
                y_batch.append(identity[y_int[i+step:i+step+SEQ_LEN]]) # Sequence-to-Sequence
            if count == BATCH_SIZE:
                count = 0
                x_batch = np.array(x_batch, dtype=np.float32)
                y_batch = np.array(y_batch, dtype=np.float32)
                yield (x_batch, y_batch)


def train(x, y, loss, step=1, pretrained=False):
    if pretrained:
        model = load_model(MODEL_FILE + '.h5', custom_objects=CUSTOM_OBJECTS)
    else:
        if CONVOLUTION:
            conv_input_layer = Input(shape=(LATITUDES+1, LONGITUDES+1, 1))
            conv_layer_0 = Conv2D(16, (2,2), activation='relu')(conv_input_layer)
            if POOLING:
                pool_layer_0 = MaxPooling2D((1,2))(conv_layer_0)
            else:
                pool_layer_0 = Conv2D(16, (1,2), strides=(1,2), activation='relu')(conv_layer_0)
            conv_layer_1 = Conv2D(32, (3,3), activation='relu')(pool_layer_0)
            if POOLING:
                pool_layer_1 = MaxPooling2D((2,2))(conv_layer_1)
            else:
                pool_layer_1 = Conv2D(32, (2,2), strides=(2,2), activation='relu')(conv_layer_1)
            conv_layer_2 = Conv2D(64, (3,3), activation='relu')(pool_layer_1)
            if POOLING:
                pool_layer_2 = MaxPooling2D((2,2))(conv_layer_2)
            else:
                pool_layer_2 = Conv2D(64, (2,2), strides=(2,2), activation='relu')(conv_layer_2)
            conv_layer_3 = Conv2D(128, (2,2), activation='relu')(pool_layer_2)
            if POOLING:
                pool_layer_3 = MaxPooling2D((2,2))(conv_layer_3)
            else:
                pool_layer_3 = Conv2D(128, (2,2), strides=(2,2), activation='relu')(conv_layer_3)
            conv_layer_4 = Conv2D(256, (3,3), activation='relu')(pool_layer_3)
            if POOLING:
                pool_layer_4 = MaxPooling2D((2,2))(conv_layer_4)
            else:
                pool_layer_4 = Conv2D(256, (2,2), strides=(2,2), activation='relu')(conv_layer_4)
            conv_layer_5 = Conv2D(512, (3,3), activation='relu')(pool_layer_4)
            if POOLING:
                pool_layer_5 = MaxPooling2D((2,2))(conv_layer_5)
            else:
                pool_layer_5 = Conv2D(512, (2,2), strides=(2,2), activation='relu')(conv_layer_5)
            flatten_layer = Flatten()(pool_layer_5)
            conv_subnet = Model(conv_input_layer, flatten_layer)
            reverse_conv_input_layer = Input(shape=(512,))
            reverse_flatten_layer = Reshape((1, 1, 512))(reverse_conv_input_layer)
            if POOLING:
                reverse_pool_layer_5 = UpSampling2D((2,2))(reverse_flatten_layer)
            else:
                reverse_pool_layer_5 = Conv2DTranspose(512, (2,2), strides=(2,2), activation='relu')(reverse_flatten_layer)
            reverse_conv_layer_5 = Conv2DTranspose(256, (3,3), activation='relu')(reverse_pool_layer_5)
            if POOLING:
                reverse_pool_layer_4 = UpSampling2D((2,2))(reverse_conv_layer_5)
            else:
                reverse_pool_layer_4 = Conv2DTranspose(256, (2,2), strides=(2,2), activation='relu')(reverse_conv_layer_5)
            reverse_conv_layer_4 = Conv2DTranspose(128, (3,3), activation='relu')(reverse_pool_layer_4)
            if POOLING:
                reverse_pool_layer_3 = UpSampling2D((2,2))(reverse_conv_layer_4)
            else:
                reverse_pool_layer_3 = Conv2DTranspose(128, (2,2), strides=(2,2), activation='relu')(reverse_conv_layer_4)
            reverse_conv_layer_3 = Conv2DTranspose(64, (2,2), activation='relu')(reverse_pool_layer_3)
            if POOLING:
                reverse_pool_layer_2 = UpSampling2D((2,2))(reverse_conv_layer_3)
            else:
                reverse_pool_layer_2 = Conv2DTranspose(64, (2,2), strides=(2,2), activation='relu')(reverse_conv_layer_3)
            reverse_conv_layer_2 = Conv2DTranspose(32, (3,3), activation='relu')(reverse_pool_layer_2)
            if POOLING:
                reverse_pool_layer_1 = UpSampling2D((2,2))(reverse_conv_layer_2)
            else:
                reverse_pool_layer_1 = Conv2DTranspose(32, (2,2), strides=(2,2), activation='relu')(reverse_conv_layer_2)
            reverse_conv_layer_1 = Conv2DTranspose(16, (3,3), activation='relu')(reverse_pool_layer_1)
            if POOLING:
                reverse_pool_layer_0 = UpSampling2D((1,2))(reverse_conv_layer_1)
            else:
                reverse_pool_layer_0 = Conv2DTranspose(16, (1,2), strides=(1,2), activation='relu')(reverse_conv_layer_1)
            reverse_conv_layer_0 = Conv2DTranspose(1, (2,2), activation='linear')(reverse_pool_layer_0)
            reverse_conv_subnet = Model(reverse_conv_input_layer, reverse_conv_layer_0)
        if STATEFUL:
            if PREDICTION_MODE == 'regression':
                if CONVOLUTION:
                    input_layer = Input(batch_shape=(BATCH_SIZE, SEQ_LEN, LATITUDES+1, LONGITUDES+1, 1))
                    conv_net = TimeDistributed(conv_subnet)(input_layer)
                else:
                    input_layer = Input(batch_shape=(BATCH_SIZE, SEQ_LEN, (LATITUDES+1) * (LONGITUDES+1)))
                    conv_net = input_layer
                #projection_layer = TimeDistributed(Dense(HIDDEN_NODES, activation='linear'))(conv_net)
                hidden_layer_1 = LSTM(HIDDEN_NODES, name='lstm1', activation=HIDDEN_ACT, recurrent_activation=GATE_ACT, return_sequences=True, stateful=True, dropout=DROPOUT_PROB, recurrent_dropout=DROPOUT_PROB, implementation=1)(conv_net)
                #residual_1 = add([hidden_layer_1, projection_layer])
                hidden_layer_2 = LSTM(HIDDEN_NODES, name='lstm2', activation=HIDDEN_ACT, recurrent_activation=GATE_ACT, return_sequences=True, stateful=True, dropout=DROPOUT_PROB, recurrent_dropout=DROPOUT_PROB, implementation=1)(concatenate([hidden_layer_1, conv_net]))
                residual_2 = add([hidden_layer_2, hidden_layer_1])
                hidden_layer_3 = LSTM(HIDDEN_NODES, name='lstm3', activation=HIDDEN_ACT, recurrent_activation=GATE_ACT, return_sequences=True, stateful=True, dropout=DROPOUT_PROB, recurrent_dropout=DROPOUT_PROB, implementation=1)(concatenate([residual_2, hidden_layer_2, hidden_layer_1, conv_net]))
                residual_3 = add([hidden_layer_3, residual_2])
                hidden_layer_4 = LSTM(HIDDEN_NODES, name='lstm4', activation=HIDDEN_ACT, recurrent_activation=GATE_ACT, return_sequences=True, stateful=True, dropout=DROPOUT_PROB, recurrent_dropout=DROPOUT_PROB, implementation=1)(concatenate([residual_3, hidden_layer_3, hidden_layer_2, hidden_layer_1, conv_net]))
                residual_4 = add([hidden_layer_4, residual_3])
                hidden_layer_5 = LSTM(HIDDEN_NODES, name='lstm5', activation=HIDDEN_ACT, recurrent_activation=GATE_ACT, return_sequences=True, stateful=True, dropout=DROPOUT_PROB, recurrent_dropout=DROPOUT_PROB, implementation=1)(concatenate([residual_4, hidden_layer_4, hidden_layer_3, hidden_layer_2, hidden_layer_1, conv_net]))
                #residual_5 = add([hidden_layer_5, residual_4])
                if CONVOLUTION:
                    dense_layer = TimeDistributed(Dense(512, activation='relu'))(hidden_layer_5)
                    output_layer = TimeDistributed(reverse_conv_subnet)(dense_layer)
                else:
                    output_layer = TimeDistributed(Dense((LATITUDES+1) * (LONGITUDES+1), activation=OUTPUT_ACT))(hidden_layer_5)
            elif PREDICTION_MODE == 'classification':
                input_layer = Input(batch_shape=(BATCH_SIZE, SEQ_LEN, (LATITUDES+1) * (LONGITUDES+1), ONEHOT_LENGTH))
                hidden_layer_1 = TimeDistributed(LSTM(HIDDEN_NODES, name='lstm1', activation=HIDDEN_ACT, recurrent_activation=GATE_ACT, return_sequences=True, stateful=True, dropout=DROPOUT_PROB, recurrent_dropout=DROPOUT_PROB, implementation=1))(conv_net)
                hidden_layer_2 = TimeDistributed(LSTM(HIDDEN_NODES, name='lstm2', activation=HIDDEN_ACT, recurrent_activation=GATE_ACT, return_sequences=True, stateful=True, dropout=DROPOUT_PROB, recurrent_dropout=DROPOUT_PROB, implementation=1))(concatenate([hidden_layer_1, conv_net]))
                residual_2 = add([hidden_layer_2, hidden_layer_1])
                hidden_layer_3 = TimeDistributed(LSTM(HIDDEN_NODES, name='lstm3', activation=HIDDEN_ACT, recurrent_activation=GATE_ACT, return_sequences=True, stateful=True, dropout=DROPOUT_PROB, recurrent_dropout=DROPOUT_PROB, implementation=1))(concatenate([residual_2, hidden_layer_2, hidden_layer_1, conv_net]))
                residual_3 = add([hidden_layer_3, residual_2])
                hidden_layer_4 = TimeDistributed(LSTM(HIDDEN_NODES, name='lstm4', activation=HIDDEN_ACT, recurrent_activation=GATE_ACT, return_sequences=True, stateful=True, dropout=DROPOUT_PROB, recurrent_dropout=DROPOUT_PROB, implementation=1))(concatenate([residual_3, hidden_layer_3, hidden_layer_2, hidden_layer_1, conv_net]))
                residual_4 = add([hidden_layer_4, residual_3])
                hidden_layer_5 = TimeDistributed(LSTM(HIDDEN_NODES, name='lstm5', activation=HIDDEN_ACT, recurrent_activation=GATE_ACT, return_sequences=True, stateful=True, dropout=DROPOUT_PROB, recurrent_dropout=DROPOUT_PROB, implementation=1))(concatenate([residual_4, hidden_layer_4, hidden_layer_3, hidden_layer_2, hidden_layer_1, conv_net]))
                output_layer = TimeDistributed(TimeDistributed(Dense(ONEHOT_LENGTH, activation='softmax')))(hidden_layer_5)
        else:
            x = x[:len(x)/BATCH_SIZE*BATCH_SIZE]
            y = y[:len(y)/BATCH_SIZE*BATCH_SIZE]
            valid_split_point = int(len(x) * 0.95)/BATCH_SIZE*BATCH_SIZE
            trains = (valid_split_point - SEQ_LEN - 1) / BATCH_SIZE
            valids = (len(x) - valid_split_point - SEQ_LEN - 1) / BATCH_SIZE
            x_train = x[:valid_split_point]
            y_train = y[:valid_split_point]
            x_valid = x[valid_split_point:]
            y_valid = y[valid_split_point:]
            if PREDICTION_MODE == 'regression':
                if CONVOLUTION:
                    input_layer = Input(shape=(SEQ_LEN, LATITUDES+1, LONGITUDES+1, 1))
                    conv_net = TimeDistributed(conv_subnet)(input_layer)
                else:
                    input_layer = Input(shape=(SEQ_LEN, (LATITUDES+1) * (LONGITUDES+1))) 
                    conv_net = input_layer
                #stochastic_timeskip_layer = StochasticTimeskip(TIMESKIP_PROB)(input_layer)
                hidden_layer_1 = LSTM(HIDDEN_NODES, name='lstm1', activation=HIDDEN_ACT, recurrent_activation=GATE_ACT, return_sequences=True, dropout=DROPOUT_PROB, recurrent_dropout=DROPOUT_PROB, implementation=1)(conv_net)
                hidden_layer_2 = LSTM(HIDDEN_NODES, name='lstm2', activation=HIDDEN_ACT, recurrent_activation=GATE_ACT, return_sequences=True, dropout=DROPOUT_PROB, recurrent_dropout=DROPOUT_PROB, implementation=1)(concatenate([hidden_layer_1, conv_net]))
                residual_2 = add([hidden_layer_2, hidden_layer_1])
                hidden_layer_3 = LSTM(HIDDEN_NODES, name='lstm3', activation=HIDDEN_ACT, recurrent_activation=GATE_ACT, return_sequences=True, dropout=DROPOUT_PROB, recurrent_dropout=DROPOUT_PROB, implementation=1)(concatenate([residual_2, hidden_layer_2, hidden_layer_1, conv_net]))
                residual_3 = add([hidden_layer_3, residual_2])
                hidden_layer_4 = LSTM(HIDDEN_NODES, name='lstm4', activation=HIDDEN_ACT, recurrent_activation=GATE_ACT, return_sequences=True, dropout=DROPOUT_PROB, recurrent_dropout=DROPOUT_PROB, implementation=1)(concatenate([residual_3, hidden_layer_3, hidden_layer_2, hidden_layer_1, conv_net]))
                residual_4 = add([hidden_layer_4, residual_3])
                hidden_layer_5 = LSTM(HIDDEN_NODES, name='lstm5', activation=HIDDEN_ACT, recurrent_activation=GATE_ACT, return_sequences=True, dropout=DROPOUT_PROB, recurrent_dropout=DROPOUT_PROB, implementation=1)(concatenate([residual_4, hidden_layer_4, hidden_layer_3, hidden_layer_2, hidden_layer_1, conv_net]))
                if CONVOLUTION:
                    dense_layer = TimeDistributed(Dense(512, activation='relu'))(hidden_layer_5)
                    output_layer = TimeDistributed(reverse_conv_subnet)(dense_layer)
                else:
                    output_layer = TimeDistributed(Dense((LATITUDES+1) * (LONGITUDES+1), activation=OUTPUT_ACT))(hidden_layer_5)
            elif PREDICTION_MODE == 'classification':
                input_layer = Input(shape=(SEQ_LEN, (LATITUDES+1) * (LONGITUDES+1), ONEHOT_LENGTH))
                stochastic_timeskip_layer = TimeDistributed(StochasticTimeskip(TIMESKIP_PROB))(input_layer)
                hidden_layer_1 = TimeDistributed(LSTM(HIDDEN_NODES, name='lstm1', activation=HIDDEN_ACT, recurrent_activation=GATE_ACT, return_sequences=True, dropout=DROPOUT_PROB, recurrent_dropout=DROPOUT_PROB, implementation=1))(conv_net)
                hidden_layer_2 = TimeDistributed(LSTM(HIDDEN_NODES, name='lstm2', activation=HIDDEN_ACT, recurrent_activation=GATE_ACT, return_sequences=True, dropout=DROPOUT_PROB, recurrent_dropout=DROPOUT_PROB, implementation=1))(concatenate([hidden_layer_1, conv_net]))
                residual_2 = add([hidden_layer_2, hidden_layer_1])
                hidden_layer_3 = TimeDistributed(LSTM(HIDDEN_NODES, name='lstm3', activation=HIDDEN_ACT, recurrent_activation=GATE_ACT, return_sequences=True, dropout=DROPOUT_PROB, recurrent_dropout=DROPOUT_PROB, implementation=1))(concatenate([residual_2, hidden_layer_2, hidden_layer_1, conv_net]))
                residual_3 = add([hidden_layer_3, residual_2])
                hidden_layer_4 = TimeDistributed(LSTM(HIDDEN_NODES, name='lstm4', activation=HIDDEN_ACT, recurrent_activation=GATE_ACT, return_sequences=True, dropout=DROPOUT_PROB, recurrent_dropout=DROPOUT_PROB, implementation=1))(concatenate([residual_3, hidden_layer_3, hidden_layer_2, hidden_layer_1, conv_net]))
                residual_4 = add([hidden_layer_4, residual_3])
                hidden_layer_5 = TimeDistributed(LSTM(HIDDEN_NODES, name='lstm5', activation=HIDDEN_ACT, recurrent_activation=GATE_ACT, return_sequences=True, dropout=DROPOUT_PROB, recurrent_dropout=DROPOUT_PROB, implementation=1))(concatenate([residual_4, hidden_layer_4, hidden_layer_3, hidden_layer_2, hidden_layer_1, conv_net]))
                output_layer = TimeDistributed(TimeDistributed(Dense(ONEHOT_LENGTH, activation='softmax')))(hidden_layer_5)
        model = Model(input_layer, output_layer)
        rmsprop_optimizer = 'rmsprop' #RMSprop(scalenorm=1.0)
        model.compile(rmsprop_optimizer, loss)
    nanstopper = TerminateOnNaN()
    if STATEFUL:
        model.fit_generator(data_generator(x, y, step), steps_per_epoch=len(x)-step, epochs=EPOCHS, callbacks=[nanstopper])
        model.save(MODEL_FILE + '.h5')
        save_state(model, STATE_FILE + '.npy')
    else:
        checkpoint = ModelCheckpoint(MODEL_FILE + '.h5', save_best_only=True)
        earlystop = EarlyStopping()
        #model.fit(x, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.05, callbacks=[checkpoint, earlystop])
        model.fit_generator(data_generator(x_train, y_train, step), steps_per_epoch=trains, epochs=EPOCHS, validation_data=data_generator(x_valid, y_valid, step), validation_steps=valids, callbacks=[checkpoint, earlystop, nanstopper])
    del model
    gc.collect()
    
def test(x, y):
    model = load_model(MODEL_FILE + '.h5', custom_objects=CUSTOM_OBJECTS)
    if STATEFUL:
        model = load_state(model, STATE_FILE + '.npy')
        predictions = []
        save_state(model, STATE_FILE + '_tested.npy')
        for data, target in zip(x, y):
            predictions.append(model.predict(np.array([data]), batch_size=BATCH_SIZE)[0][0])
            model = load_state(model, STATE_FILE + '_tested.npy')
            model.fit(np.array([data]), np.array([target]), batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=False)
            save_state(model, STATE_FILE + '_tested.npy')
        model.save(MODEL_FILE + '_tested.h5')
    else:
        raw_predictions = model.predict(np.array(x), batch_size=BATCH_SIZE)
        predictions = raw_predictions[:,-1]
    if PREDICTION_MODE == 'classification':
        predictions = np.argmax(predictions, axis=-1)
    if OUTPUT_ACT == 'sigmoid':
        y = descale(y)
        predictions = descale(predictions)
    collect_magnitudes(predictions, 'Prediction')
    for test_magnitude_cutoff in [-1.0, 5.0]:
        true_positive = 0.0
        true_negative = 0.0
        false_positive = 0.0
        false_negative = 0.0
        if CONVOLUTION:
            for i in range(len(predictions)):
                for j in range(LATITUDES+1):
                    for k in range(LONGITUDES+1):
                        true_magnitude = y[i][j][k]
                        pred_magnitude = predictions[i][j][k]
                        if true_magnitude >= test_magnitude_cutoff or pred_magnitude >= test_magnitude_cutoff:
                            #coordinate_error = ((true_latitude - pred_latitude) ** 2 + (true_longitude - pred_longitude) ** 2) ** (1.0/2.0)
                            magnitude_error = abs(true_magnitude - pred_magnitude)
                            if magnitude_error < TEST_MAGNITUDE_RANGE and true_magnitude > PREDICTION_MAGNITUDE_CUTOFF: #and coordinate_error <= TEST_COORDINATE_RANGE
                                true_positive += 1
                            elif magnitude_error < TEST_MAGNITUDE_RANGE and true_magnitude <= PREDICTION_MAGNITUDE_CUTOFF: #and coordinate_error <= TEST_COORDINATE_RANGE 
                                true_negative += 1
                            elif magnitude_error >= TEST_MAGNITUDE_RANGE and true_magnitude < pred_magnitude: #or coordinate_error >= TEST_COORDINATE_RANGE
                                false_positive += 1
                            elif magnitude_error >= TEST_MAGNITUDE_RANGE and true_magnitude >= pred_magnitude: #or coordinate_error >= TEST_COORDINATE_RANGE
                                false_negative += 1
        else:
            for i in range(len(predictions)):
                for j in range((LATITUDES+1)*(LONGITUDES+1)):
                    true_magnitude = y[i][0][j]
                    pred_magnitude = predictions[i][j]
                    if true_magnitude >= test_magnitude_cutoff or pred_magnitude >= test_magnitude_cutoff:
                        #coordinate_error = ((true_latitude - pred_latitude) ** 2 + (true_longitude - pred_longitude) ** 2) ** (1.0/2.0)
                        magnitude_error = abs(true_magnitude - pred_magnitude)
                        if magnitude_error < TEST_MAGNITUDE_RANGE and true_magnitude > PREDICTION_MAGNITUDE_CUTOFF: #and coordinate_error <= TEST_COORDINATE_RANGE
                            true_positive += 1
                        elif magnitude_error < TEST_MAGNITUDE_RANGE and true_magnitude <= PREDICTION_MAGNITUDE_CUTOFF: #and coordinate_error <= TEST_COORDINATE_RANGE 
                            true_negative += 1
                        elif magnitude_error >= TEST_MAGNITUDE_RANGE and true_magnitude < pred_magnitude: #or coordinate_error >= TEST_COORDINATE_RANGE
                            false_positive += 1
                        elif magnitude_error >= TEST_MAGNITUDE_RANGE and true_magnitude >= pred_magnitude: #or coordinate_error >= TEST_COORDINATE_RANGE
                            false_negative += 1
        total_pop = true_positive + true_negative + false_positive + false_negative
        positive = true_positive + false_negative
        negative = true_negative + false_positive
        predict_positive = true_positive + false_positive
        predict_negative = true_negative + false_negative
        if total_pop > 0:
            prevalence = positive / total_pop
            PRE = positive / total_pop # Prevalence
            ACC = (true_positive + true_negative) / total_pop # Accuracy
        else:
            prevalence = 0
            PRE = 0
            ACC = 0
        if positive > 0:
            TPR = true_positive / positive # True Positive Rate OR Sensitivity OR Recall
            FNR = false_negative / positive # False Negative Rate OR Miss Rate
        else:
            TPR = 0
            FNR = 0
        if negative > 0:
            FPR = false_positive / negative # False Positive Rate OR Fall-Out
            TNR = true_negative / negative # True Negative Rate OR Specificity
        else:
            FPR = 0
            TNR = 0
        if predict_positive > 0:
            PPV = true_positive / predict_positive # Positive Predictive Value OR Precision
            FDR = false_positive / predict_positive # False Discovery Rate
        else:
            PPV = 0
            FDR = 0
        if predict_negative > 0:
            FOR = false_negative / predict_negative # False Omission Rate
            NPV = true_negative / predict_negative # Negative Predictive Value
        else:
            FOR = 0
            NPV = 0
        if FPR > 0:
            PLR = TPR / FPR # Positive Likelihood Ratio
        else:
            PLR = 0
        if TNR > 0:
            NLR = FNR / TNR # Negative Likelihood Ratio
        else:
            NLR = 0
        if NLR > 0:
            DOR = PLR / NLR # Diagnostic Odds Ratio
        else:
            DOR = 0
        if (true_positive + false_positive + false_negative) > 0:
            J = true_positive / (true_positive + false_positive + false_negative) # Jaccard Index
        else:
            J = 0
        if (PPV + TPR) > 0:
            F1 = 2 * ((PPV * TPR) / (PPV + TPR))
        else:
            F1 = 0
        if true_positive > 0 and true_negative > 0 and false_positive > 0 and false_negative > 0:
            MCC = (true_positive * true_negative - false_positive * false_negative) / ((true_positive + false_positive) * (true_positive + false_negative) * (true_negative + false_positive) * (true_negative + false_negative)) ** 0.5
        else:
            MCC = 0
        BM = TPR + TNR - 1
        MK = PPV + NPV - 1
        BA = (TPR + TNR) / 2
        with open(TEST_FILE + '_tested.txt', 'a') as f:
            f.write('Magnitude = ' + str(test_magnitude_cutoff) + '\n')
            f.write('True Positive (TP) = ' + str(true_positive) + '\n')
            f.write('True Negative (TN) = ' + str(true_negative) + '\n')
            f.write('False Positive (FP) = ' + str(false_positive) + '\n')
            f.write('False Negative (FN) = ' + str(false_negative) + '\n')
            f.write('Total Population (TP + TN + FP + FN) = ' + str(total_pop) + '\n')
            f.write('Positive (TP + FN) = ' + str(positive) + '\n')
            f.write('Negative (FP + TN) = ' + str(negative) + '\n')
            f.write('Predict Positive (TP) = ' + str(predict_positive) + '\n')
            f.write('Predict Negative (TP) = ' + str(predict_negative) + '\n')
            f.write('Prevalence (PRE = positive / total_pop) = ' + str(PRE) + '\n')
            f.write('True Positive Rate OR Sensitivity OR Recall (TPR = TP / positive) = ' + str(TPR) + '\n')
            f.write('False Negative Rate OR Miss Rate (FNR = FN / positive) = ' + str(FNR) + '\n')
            f.write('False Positive Rate OR Fall-Out (FPR = FP / negative) = ' + str(FPR) + '\n')
            f.write('True Negative Rate OR Specificity (TNR = TN / negative) = ' + str(TNR) + '\n')
            f.write('Accuracy (ACC = (TP + TN) / total_pop) = ' + str(ACC) + '\n')
            f.write('Positive Predictive Value OR Precision (PPV = TP / predict_positive) = ' + str(PPV) + '\n')
            f.write('False Discovery Rate (FDR = FP / predict_positive) = ' + str(FDR) + '\n')
            f.write('False Omission Rate (FOR = FN / predict_negative) = ' + str(FOR) + '\n')
            f.write('Negative Predictive Value (NPV = TN / predict_negative) = ' + str(NPV) + '\n')
            f.write('Positive Likelihood Ratio (PLR = TPR / FPR) = ' + str(PLR) + '\n')
            f.write('Negative Likelihood Ratio (NLR = FNR / TNR) = ' + str(NLR) + '\n')
            f.write('Diagnostic Odds Ratio (DOR = PLR / NLR) = ' + str(DOR) + '\n')
            f.write('Jaccard Index (IoU) (J = TP / (TP + FP + FN)) = ' + str(J) + '\n')
            f.write('F1-Score = ' + str(F1) + '\n')
            f.write('Matthews Correlation Coefficient = ' + str(MCC) + '\n')
            f.write('Informedness = ' + str(BM) + '\n')
            f.write('Markedness = ' + str(MK) + '\n')
            f.write('Balanced Accuracy = ' + str(BA) + '\n')
    del model
    gc.collect()
    
def collect_magnitudes(data, label):
    magnitudes = {}
    for i in frange(-1.0, 10.0, 0.5):
        magnitudes[i] = 0
    if CONVOLUTION:
        for i in range(int(TEST_TOTAL_TIME / SECONDS_PER_TIMESTEP)-SEQ_LEN):
            for j in range(LATITUDES+1):
                for k in range(LONGITUDES+1):
                    for l in frange(-1.0, 10.0, 0.5):
                        magnitude = data[i][j][k]
                        if magnitude >= l and magnitude < l + 0.5:
                            magnitudes[l] += 1
    else:
        for i in range(int(TEST_TOTAL_TIME / SECONDS_PER_TIMESTEP)-SEQ_LEN):
            for j in range((LATITUDES+1)*(LONGITUDES+1)):
                for k in frange(-1.0, 10.0, 0.5):
                    magnitude = data[i][j]
                    if magnitude >= k and magnitude < k + 0.5:
                        magnitudes[k] += 1
    with open(TEST_FILE + '_tested.txt', 'a') as f:
        f.write(label + ' Magnitudes:\n')
        for i in frange(-1.0, 10.0, 0.5):
            f.write(str(i) + '-' + str(i+0.5) + ': ' + str(magnitudes[i]) + '\n')        

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def update(x, y):
    if STATEFUL:
        model = load_model(MODEL_FILE + '_tested.h5', custom_objects=CUSTOM_OBJECTS)
        model = load_state(model, STATE_FILE + '_tested.npy')
        model.fit(np.array(x), np.array(y), batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=False)
        model.save(MODEL_FILE + '_updated.h5')
        save_state(model, STATE_FILE + '_updated.npy')
        model.save(MODEL_FILE + '_live.h5')
        save_state(model, STATE_FILE + '_live.npy')
    del model
    gc.collect()

def predict(x, y, models, today=None):
    if today is None:
        today = datetime.utcnow()
        today.replace(hour=0, minute=0, second=0, microsecond=0)
    date = str(today.year) + '-' + pad_number(today.month) + '-' + pad_number(today.day)
    for name in models:
        for model_name, model_file, state_file in zip(MODEL_NAMES, MODEL_FILES, STATE_FILES):
            if name == model_name:
                model = load_model(model_file + '_live.h5', custom_objects=CUSTOM_OBJECTS)
                if STATEFUL:
                    model = load_state(model, state_file + '_live.npy')
                    model.fit(np.array([x]), np.array([y]), batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=False)
                    save_state(model, state_file + '_live.npy')
                    predictions = model.predict(np.array([y]), batch_size=BATCH_SIZE)
                    model.save(model_file + '_live.h5')
                    raw_predictions = np.array(predictions)
                else:
                    raw_predictions = model.predict(np.array([y]), batch_size=BATCH_SIZE)[:,-1]
                if OUTPUT_ACT == 'sigmoid':
                    raw_predictions = descale(raw_predictions)
                predictions = np.where(raw_predictions > PREDICTION_MAGNITUDE_CUTOFF, raw_predictions, PREDICTION_MAGNITUDE_CUTOFF).reshape((LATITUDES+1, LONGITUDES+1))
                Image.fromarray(((predictions / 10) * 255).astype(np.int32)).save('prediction/' + model_name + '/' + date + '.png')
                with open('prediction/' + model_name + '/' + date + '.csv', 'w') as f:
                    for i in range(LATITUDES+1):
                        for j in range(LONGITUDES+1):
                            prediction = predictions[i][j]
                            if prediction > PREDICTION_MAGNITUDE_CUTOFF:
                                f.write(str(i * 2 - LATITUDES) + ', ' + str(j * 2 - LONGITUDES) + ', '+ str(prediction) + '\n')
                del model
                gc.collect()

def save_state(model, file_path):
    states = []
    for layer_name in STATEFUL_LAYER_NAMES:
        states.append(model.get_layer(layer_name).states)
    np.save(file_path, np.array(states))

def load_state(model, file_path):
    states = np.load(file_path, allow_pickle=True)
    for i, layer_name in enumerate(STATEFUL_LAYER_NAMES):
        model.get_layer(layer_name).reset_states(states=[states[i][0].numpy(), states[i][1].numpy()])
    return model

def main():
    download_dataset()
    download_update()

    generate_dataset()
    generate_testset()
    generate_update()
    generate_seedset(TODAY)

    with open('date.txt', 'w') as f:
        f.write(UPDATE_END_DATE.strftime('%Y-%m-%d'))

    global MODEL_FILE
    global STATE_FILE
    global TEST_FILE
    global LOSS

    for MODEL_FILE, STATE_FILE, TEST_FILE, LOSS in zip(MODEL_FILES, STATE_FILES, TEST_FILES, LOSSES):
        x, y = load_dataset()
        train(x, y, LOSS)
        x, y = load_testset()
        test(x, y)
        x, y = load_update()
        update(x, y)
    

if __name__ == "__main__":
    main()
    