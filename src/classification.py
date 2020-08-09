# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 16:28:18 2020

@author: eliphat

   Copyright 2020 Shanghai Jiao Tong University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""
import keras
import numpy
import acae
import data_flower
import vnapf
import acae_kis4
from joblib import Parallel, delayed


def proposed_hard_sampling(pts, prob, k=8):
    # pts: (n, 3)
    # prob: (4, n)
    assert pts.shape[0] == prob.shape[1]
    n = pts.shape[0]
    picked = [pts[prob.argmax() % n]]
    nms = [pts[prob.argmax() % n]]
    while 1:
        for j in range(len(prob)):
            for i, arg in enumerate(numpy.argsort(prob[j])[::-1]):
                p = pts[arg % n]
                if i >= n // 5:
                    if min(numpy.sum((p - po) ** 2) for po in picked) < 0.03:
                        continue
                    picked.append(p)
                    nms.append(p)
                    if len(picked) == k:
                        return numpy.array(picked)
                    break
                if min(numpy.sum((p - po) ** 2) for po in nms) < 0.03:
                    nms.append(p)
                    continue  # NMS
                picked.append(p)
                if len(picked) == k:
                    return numpy.array(picked)
                nms.append(p)
                break


def parallel_hard_sampling(ptsc, probs, k=8):
    r = Parallel(8, verbose=1)(
            delayed(proposed_hard_sampling)(pts, prob, k)
            for pts, prob in zip(ptsc, probs)
        )
    return numpy.array(r)


def dense_block(x, size):
    y = keras.layers.Dense(size)(x)
    y_bn = keras.layers.BatchNormalization()(y)
    y_a = keras.layers.ReLU()(y_bn)
    return y_a


def mlp_block(x, *sizes):
    for size in sizes:
        x = dense_block(x, size)
    return x


def cla_pointnet():

    def _call(x):
        y1 = mlp_block(x, 64, 64)
        yc2 = mlp_block(y1, 64, 128, 1024)
        yc_p = keras.layers.GlobalMaxPool1D()(yc2)
        yc = mlp_block(yc_p, 512, 256, 40)
        yc_a = keras.layers.Softmax(name='aux')(yc)
        return yc_a

    return _call


def network_8pt():
    x = keras.layers.Input([8, 3])
    y = cla_pointnet()(x)
    return keras.models.Model(inputs=x, outputs=y)


def farthest_sampling(x, cnt=8):
    # x: n x m x 3
    p = numpy.zeros([x.shape[0], cnt, 3])
    for i in range(cnt):
        centroid = numpy.mean(x, axis=-2, keepdims=True)  # n x 1 x 3
        dst = numpy.sum((x - centroid) ** 2, axis=-1)  # n x m
        farthest = numpy.argmax(dst, axis=-1)  # n
        f_expand = numpy.expand_dims(farthest, axis=-1)  # n x 1
        ranged = numpy.arange(0, x.shape[1])
        r_expand = numpy.expand_dims(ranged, axis=0)  # 1 x m
        selection = numpy.equal(f_expand, r_expand)
        discarded = numpy.not_equal(f_expand, r_expand)
        p[:, i, :] = x[selection]
        x = x[discarded].reshape([x.shape[0], x.shape[1] - 1, 3])
    return p


def train_acae_hard(save_samples_only=False):
    x, xl = data_flower.all_h5(vnapf.DATASET, True, True)  # n x 2048 x 3
    x_test, xl_test = data_flower.all_h5(vnapf.TESTSET, True, True)
    enc, model = acae_kis4.network(return_enc=True)
    model.load_weights("weights_acae_k4.h5")
    _, prob = enc.predict(x, verbose=1)
    _, prob_t = enc.predict(x_test, verbose=1)
    kp = parallel_hard_sampling(x, prob)
    kp_t = parallel_hard_sampling(x_test, prob_t)
    if save_samples_only:
        numpy.save("kp_t_kis4.npy", kp_t)
        return
    model = network_8pt()
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["acc"]
    )
    clbk = [keras.callbacks.CSVLogger("8pt_acae_k4.csv")]
    model.fit(x=kp, y=xl, batch_size=64, epochs=50,
              validation_data=(kp_t, xl_test),
              callbacks=clbk)


def train_acae():
    x, xl = data_flower.all_h5(vnapf.DATASET, True, True)  # n x 2048 x 3
    x_test, xl_test = data_flower.all_h5(vnapf.TESTSET, True, True)
    enc, model = acae.network(return_enc=True)
    model.load_weights("weights_acae.h5")

    kp, prob = enc.predict(x, verbose=1)
    am = numpy.argsort(prob, axis=-1)[:, :, -1:].reshape([kp.shape[0], -1])
    kp = numpy.zeros([am.shape[0], am.shape[1], 3])

    kp_t, prob_t = enc.predict(x_test, verbose=1)
    am_t = numpy.argsort(prob_t, axis=-1)[:, :, -1:].reshape([kp_t.shape[0], -1])
    kp_t = numpy.zeros([am_t.shape[0], am_t.shape[1], 3])
    for i in range(len(x_test)):
        for j in range(len(am_t[i])):
            kp_t[i][j] = x_test[i][am_t[i][j]]
    for i in range(len(x)):
        for j in range(len(am[i])):
            kp[i][j] = x[i][am[i][j]]

    model = network_8pt()
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["acc"]
    )
    clbk = [keras.callbacks.CSVLogger("8pt_acae.csv")]
    model.fit(x=kp, y=xl, batch_size=64, epochs=50,
              validation_data=(kp_t, xl_test),
              callbacks=clbk)


def train_vnapf():
    x, xl = data_flower.all_h5(vnapf.DATASET, True, True)  # n x 2048 x 3
    x_test, xl_test = data_flower.all_h5(vnapf.TESTSET, True, True)
    enc, model = vnapf.network(return_enc=True)
    model.load_weights("weights.h5")

    kp, prob = enc.predict(x, verbose=1)
    am = numpy.argsort(prob, axis=-1)[:, :, -1:].reshape([kp.shape[0], -1])
    kp = numpy.zeros([am.shape[0], am.shape[1], 3])

    kp_t, prob_t = enc.predict(x_test, verbose=1)
    am_t = numpy.argsort(prob_t, axis=-1)[:, :, -1:].reshape([kp_t.shape[0], -1])
    kp_t = numpy.zeros([am_t.shape[0], am_t.shape[1], 3])
    for i in range(len(x_test)):
        for j in range(len(am_t[i])):
            kp_t[i][j] = x_test[i][am_t[i][j]]
    for i in range(len(x)):
        for j in range(len(am[i])):
            kp[i][j] = x[i][am[i][j]]
    model = network_8pt()
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["acc"]
    )
    clbk = [keras.callbacks.CSVLogger("8pt_naive.csv")]
    model.fit(x=kp, y=xl, batch_size=64, epochs=50,
              validation_data=(kp_t, xl_test),
              callbacks=clbk)


def train_fps():
    x, xl = data_flower.all_h5(vnapf.DATASET, True, True)  # n x 2048 x 3
    x_test, xl_test = data_flower.all_h5(vnapf.TESTSET, True, True)
    kp_train = farthest_sampling(x)
    kp_test = farthest_sampling(x_test)
    model = network_8pt()
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["acc"]
    )
    clbk = [keras.callbacks.CSVLogger("8pt_fps.csv")]
    model.fit(x=kp_train, y=xl, batch_size=64, epochs=50,
              validation_data=(kp_test, xl_test),
              callbacks=clbk)
