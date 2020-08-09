# -*- coding: utf-8 -*-
"""
Basic KAE
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
import classification


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
DATASET = r"./point_cloud/train"
TESTSET = r"D:/AI_DataSet/point_cloud/test"


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
    x = keras.layers.Input([k, point_dims])
    y1 = mlp_block(x, 64, 64)
    y1_r = keras.layers.Flatten()(y1)
    y2 = mlp_block(y1_r, 256)
    yf = keras.layers.Dense(1024 * point_dims)(y2)
    yf_r = keras.layers.Reshape([1024, 3])(yf)
    return keras.models.Model(inputs=x, outputs=yf_r)


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
    y_enc, prob = enc(x)
    dec = decoder(8, point_dims)
    y_dec = dec(y_enc)
    if return_enc:
        return enc, keras.models.Model(inputs=x, outputs=y_dec)
    return keras.models.Model(inputs=x, outputs=y_dec)


def repeatability_test():
    x_test = data_flower.all_h5(TESTSET, True)
    enc, model = network(return_enc=True)
    model.load_weights("weights.h5")
    kp, prob = enc.predict(x_test, verbose=1)
    rand = numpy.random.normal(size=[3, 1])
    rand_T = numpy.transpose(rand)
    gemm = numpy.matmul
    ortho = numpy.eye(3) - 2 * (gemm(rand, rand_T)) / (gemm(rand_T, rand))
    x_test_r = numpy.dot(x_test, ortho)
    kp_r, prob_r = enc.predict(x_test_r, verbose=1)
    am = numpy.argmax(prob, axis=-1)
    am_r = numpy.argmax(prob_r, axis=-1)
    for i in range(len(x_test)):
        for j in range(len(am[i])):
            kp[i][j] = x_test[i][am[i][j]]
            kp_r[i][j] = x_test_r[i][am_r[i][j]]
    kp_r = numpy.expand_dims(kp_r, axis=-2)  # k1 x 1 x 3
    kp = numpy.dot(kp, ortho)
    kp = numpy.expand_dims(kp, axis=-3)  # 1 x k2 x 3
    delta = numpy.sum(numpy.square(kp - kp_r), axis=-1)
    nearest = numpy.min(delta, -2)
    eq = numpy.less_equal(nearest, 0.05 ** 2)
    print("Repeatability:", numpy.mean(eq))


def visual_test(reload=False):
    global x_test, enc, model
    if reload:
        x_test, x_label = data_flower.all_h5(TESTSET, True, True)
        x_test = x_test[numpy.equal(x_label, 0)[:, 0], :, :]
        enc, model = network(return_enc=True)
        model.load_weights("weights.h5")
    pick = random.choice(x_test)
    pd = enc.predict(numpy.array([pick]))[1][0]
    # global Mx
    '''Mx = enc.predict(numpy.array([pick]))[1][0]
    MpSort = numpy.argsort(Mx, axis=-1)
    Mp = numpy.reshape(MpSort[:, -2:], [-1])
    kp = enc.predict(numpy.array([pick]))[0][0]'''
    kp = classification.proposed_hard_sampling(pick, pd, 16)
    numpy.savetxt("ipc.txt", pick)
    numpy.savetxt("okp.txt", kp)
    visualizations.merge_pcd(kp, pick)
    # visualizations.show_point_cloud_array(pick)


def train():
    x = data_flower.all_h5(DATASET, True)  # n x 2048 x 3
    x_test = data_flower.all_h5(TESTSET, True)
    model = network()
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=loss_fn,
        metrics=[deviation_regularization]
    )
    clbk = [
        keras.callbacks.CSVLogger("training.csv"),
        keras.callbacks.ModelCheckpoint("weights.h5",
                                        save_best_only=True,
                                        save_weights_only=True)
    ]
    model.fit(x=x, y=x, batch_size=64, epochs=50,
              validation_data=(x_test, x_test),
              callbacks=clbk)
