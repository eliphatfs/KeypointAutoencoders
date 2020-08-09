# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 10:01:16 2020

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
import os
import random

import numpy
import tensorflow as tf
import keras.backend as K
import keras

import data_flower
import visualizations


DATASET = r"./point_cloud/train"
TESTSET = r"./point_cloud/test"


def cut_points(args, k=100):  # Hard cut, BP will fail
    points, probab = args
    argsorted = tf.argsort(probab, direction='DESCENDING')
    indices = tf.gather(argsorted, tf.range(k), axis=-1)
    return [tf.gather(points, indices, axis=-2, batch_dims=1),
            tf.gather(probab, indices, axis=-1, batch_dims=1)]


def cut_points_layer(k=100):
    return keras.layers.Lambda(lambda x: cut_points(x, k))


def dense_block(x, size):
    y = keras.layers.Dense(size)(x)
    y_bn = keras.layers.BatchNormalization()(y)
    y_a = keras.layers.ReLU()(y_bn)
    return y_a


def mlp_block(x, *sizes):
    for size in sizes:
        x = dense_block(x, size)
    return x


def repeat_global(args):
    glo, target = args
    return keras.layers.RepeatVector(K.shape(target)[-2])(glo)


def merge_global(args):
    glo, loc = args
    repeated = repeat_global([glo, loc])
    merged = K.concatenate([loc, repeated])
    return merged


def probab_merge(args):
    points, prob = args
    # points: k x 3
    # prob: 32 x k
    # out: 32 x 3
    prob = K.expand_dims(prob)  # 32 x k x 1
    points = K.expand_dims(points, axis=-3)  # 1 x k x 3
    merge = prob * points  # broadcast, 32 x k x 3
    return K.sum(merge, axis=-2)


def vec_norm(x):
    return K.sqrt(K.sum(K.square(x), axis=-1) + K.epsilon())


def encoder(n_out=32, point_dims=3):
    x = keras.layers.Input([None, point_dims])
    y1 = mlp_block(x, 64, 64)
    y2 = mlp_block(y1, 64, 128, 1024)
    y_p = keras.layers.GlobalMaxPool1D()(y2)
    merged = keras.layers.Lambda(merge_global)([y_p, y1])
    yf = mlp_block(merged, 512, 256, n_out)  # k x 32
    yf_p = keras.layers.Permute([2, 1])(yf)  # 32 x k
    yf_pa = keras.layers.Softmax()(yf_p)
    # to merge with k x 3
    yf_pts = keras.layers.Lambda(probab_merge)([x, yf_pa])
    return keras.models.Model(inputs=x, outputs=[yf_pts, yf_pa])


def decoder(k=32, point_dims=3):

    def _call(x):
        y1 = mlp_block(x, 64, 64)
        yc2 = mlp_block(y1, 64, 128, 1024)
        yc_p = keras.layers.GlobalMaxPool1D()(yc2)
        yc = mlp_block(yc_p, 512, 256, 40)
        yc_a = keras.layers.Softmax(name='aux')(yc)
        y1_r = keras.layers.Flatten()(y1)
        y2 = mlp_block(y1_r, 256, 512)
        yf = keras.layers.Dense(1024 * point_dims)(y2)
        yf_r = keras.layers.Reshape([1024, point_dims], name='recon')(yf)
        return yf_r, yc_a

    return _call


def nearest_neighbour_loss(y_true, y_pred):
    # y_true: k1 x 3
    # y_pred: k2 x 3
    y_true_rep = K.expand_dims(y_true, axis=-2)  # k1 x 1 x 3
    y_pred_rep = K.expand_dims(y_pred, axis=-3)  # 1 x k2 x 3
    # k1 x k2 x 3
    y_delta = K.sum(K.square(y_pred_rep - y_true_rep), axis=-1)
    # k1 x k2
    y_nearest = K.min(y_delta, -2)
    # k2
    b_nearest = K.min(y_delta, -1)
    # k1
    return K.mean(y_nearest) + K.mean(b_nearest)


def deviation_regularization(y_true, y_pred):
    std = K.std(y_pred, axis=-2)
    return K.mean(std)


def loss_fn(y_true, y_pred):
    return nearest_neighbour_loss(y_true, y_pred)


def network(point_dims=3, return_enc=False):
    x = keras.layers.Input([None, point_dims])
    enc = encoder(8, point_dims)
    y_enc, pa = enc(x)
    dec = decoder(8, point_dims)
    y_dec, y_lab = dec(y_enc)
    if return_enc:
        return enc, keras.models.Model(inputs=x, outputs=[y_dec, y_lab])
    return keras.models.Model(inputs=x, outputs=[y_dec, y_lab])


def nor(a):
    data = a
    nmean = numpy.mean(data, axis=-2, keepdims=True)
    nstd = numpy.std(data, axis=-2, keepdims=True)
    nstd = numpy.mean(nstd, axis=-1, keepdims=True)
    return (data - nmean) / nstd


def visual_test(reload=False):
    global x_test, enc, model
    if reload:
        x_test, x_label = data_flower.all_h5(TESTSET, True, True)
        x_test = x_test[numpy.equal(x_label, 0)[:, 0], :, :]
        enc, model = network(return_enc=True)
        model.load_weights("weights_acae.h5")
    pick = random.choice(x_test)
    pd = model.predict(numpy.array([pick]))[0][0]
    global Mx
    Mx = enc.predict(numpy.array([pick]))[1][0]
    MpSort = numpy.argsort(Mx, axis=-1)
    Mp = numpy.reshape(MpSort[:, -2:], [-1])
    kp = enc.predict(numpy.array([pick]))[0][0]
    visualizations.merge_pcd(pick[list(filter(lambda x: x not in Mp, range(2048)))], pick[Mp])
    # visualizations.show_point_cloud_array(pick)
    visualizations.show_point_cloud_array(pd)
    # visualizations.show_point_cloud_array(kp)


def repeatability_test():
    x_test = data_flower.all_h5(TESTSET, True)
    enc, model = network(return_enc=True)
    model.load_weights("weights_acae.h5")
    kp, _ = enc.predict(x_test, verbose=1)

    rand = numpy.random.normal(size=[3, 1])
    rand_T = numpy.transpose(rand)
    gemm = numpy.matmul
    ortho = numpy.eye(3) - 2 * (gemm(rand, rand_T)) / (gemm(rand_T, rand))
    x_test_r = numpy.dot(x_test, ortho)
    kp_r, _ = enc.predict(x_test_r, verbose=1)

    kp_r = numpy.expand_dims(kp_r, axis=-2)  # k1 x 1 x 3
    kp = numpy.expand_dims(kp, axis=-3)  # 1 x k2 x 3
    delta = numpy.sum(numpy.square(kp - kp_r), axis=-1)
    nearest = numpy.min(delta, -2)
    eq = numpy.less_equal(nearest, 0.05 ** 2)
    print("Repeatability:", numpy.mean(eq))


def train():
    x, xl = data_flower.all_h5(DATASET, True, True)  # n x 2048 x 3
    x_test, xl_test = data_flower.all_h5(TESTSET, True, True)
    model = network()
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss={"recon": loss_fn, "aux": "sparse_categorical_crossentropy"},
        loss_weights={"recon": 0.85, "aux": 0.15},
        metrics={"recon": deviation_regularization, "aux": "acc"}
    )
    clbk = [
        keras.callbacks.CSVLogger("training_acae.csv"),
        keras.callbacks.ModelCheckpoint("weights_acae.h5",
                                        save_best_only=True,
                                        save_weights_only=True)
    ]
    model.fit(x=x, y=[x, xl], batch_size=64, epochs=50,
              validation_data=(x_test, [x_test, xl_test]),
              callbacks=clbk)


# visual_test(True)
