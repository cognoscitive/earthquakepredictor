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

import tensorflow.keras.backend as K

GOLDEN_RATIO = (1.0 + (5.0 ** 0.5)) / 2.0
GR = GOLDEN_RATIO
GRC = GOLDEN_RATIO - 1
SILVER_RATIO = 1.0 + (2.0 ** 0.5)
SR = SILVER_RATIO

def exp_or_log(y_true, y_pred):
    return K.mean(K.switch(y_pred > y_true, K.log(y_pred - y_true + 1), K.exp(-(y_pred - y_true)) - 1), axis=-1)

def linex(y_true, y_pred):
    return K.mean(K.exp(-(y_pred - y_true)) + (y_pred - y_true) - 1, axis=-1)

def logex(y_true, y_pred):
    return K.mean(K.exp(-(y_pred - y_true)) + K.log(K.abs(y_pred - y_true) + 1.0 / K.exp(1)), axis=-1)

def abex(y_true, y_pred):
    return K.mean(K.exp(-(y_pred - y_true)) + K.abs(y_pred - y_true) - 1, axis=-1)

def exp_or_lin(y_true, y_pred):
    return K.mean(K.switch(y_pred > y_true, y_pred - y_true, K.exp(-(y_pred - y_true)) - 1), axis=-1)

def sqr_or_lin(y_true, y_pred):
    return K.mean(K.switch(y_pred > y_true, y_pred - y_true, K.square(y_pred - y_true)), axis=-1)

def pow_or_lin(y_true, y_pred):
    return K.mean(K.switch(y_pred > y_true, y_pred - y_true, K.pow(-(y_pred - y_true), np.e)), axis=-1)

def lin_or_log(y_true, y_pred):
    return K.mean(K.switch(y_pred > y_true, K.log(y_pred - y_true + 1), K.abs(y_pred - y_true)), axis=-1)

def sqr_or_sqrt(y_true, y_pred):
    return K.mean(K.switch(y_pred > y_true, K.sqrt(y_pred - y_true), K.square(y_pred - y_true)), axis=-1)

def double_or_half(y_true, y_pred):
    return K.mean(K.switch(y_pred > y_true, (y_pred - y_true) / 2.0, -(y_pred - y_true) * 2.0), axis=-1)

def mule_or_dive(y_true, y_pred):
    return K.mean(K.switch(y_pred > y_true, (y_pred - y_true) / np.e, -(y_pred - y_true) * np.e), axis=-1)

def mulgr_or_divgr(y_true, y_pred):
    return K.mean(K.switch(y_pred > y_true, (y_pred - y_true) / GR, -(y_pred - y_true) * GR), axis=-1)

def twoxp_or_logtwo(y_true, y_pred):
    return K.mean(K.switch(y_pred > y_true, K.log(y_pred - y_true + 1) / K.log(2.0), K.pow(2.0, -(y_pred - y_true)) - 1), axis=-1)

def tenxp_or_logten(y_true, y_pred):
    return K.mean(K.switch(y_pred > y_true, K.log(y_pred - y_true + 1) / K.log(10.0), K.pow(10.0, -(y_pred - y_true)) - 1), axis=-1)

def golden(y_true, y_pred):
    return K.mean(K.switch(y_pred > y_true, K.log(y_pred - y_true + 1) / K.log(GR), K.pow(GR, -(y_pred - y_true)) - 1), axis=-1)

def silver(y_true, y_pred):
    return K.mean(K.switch(y_pred > y_true, K.log(y_pred - y_true + 1) / K.log(SR), K.pow(SR, -(y_pred - y_true)) - 1), axis=-1)

def sqrt5(y_true, y_pred):
    return K.mean(K.switch(y_pred > y_true, K.log(y_pred - y_true + 1) / K.log(5.0 ** 0.5), K.pow(5.0 ** 0.5, -(y_pred - y_true)) - 1), axis=-1)

def exporlogtwo(y_true, y_pred):
    return K.mean(K.switch(y_pred > y_true, K.log(y_pred - y_true + 1) / K.log(2.0), K.exp(-(y_pred - y_true)) - 1), axis=-1)

def twoxporlog(y_true, y_pred):
    return K.mean(K.switch(y_pred > y_true, K.log(y_pred - y_true + 1), K.pow(2.0, -(y_pred - y_true)) - 1), axis=-1)

def avg2e(y_true, y_pred):
    base = (2.0 + np.e) / 2.0
    return K.mean(K.switch(y_pred > y_true, K.log(y_pred - y_true + 1) / K.log(base), K.pow(base, -(y_pred - y_true)) - 1), axis=-1)

def smooth_l1(y_true, y_pred):
    return K.mean(K.switch(K.abs(y_pred - y_true) > 1, K.abs(y_pred - y_true), K.square(y_pred - y_true)), axis=-1)

def smooth_l1_l2(y_true, y_pred):
    return K.mean(K.switch(y_pred - y_true > 1, K.abs(y_pred - y_true), K.square(y_pred - y_true)), axis=-1)

def straight_l2_l1(y_true, y_pred):
    return K.mean(K.switch(y_pred - y_true > -1, K.abs(y_pred - y_true), K.square(y_pred - y_true)), axis=-1)
