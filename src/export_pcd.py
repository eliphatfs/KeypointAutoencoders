# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 20:38:19 2020

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
import numpy
import acae
import vnapf
import pickle
import data_flower


TESTSET = r"./point_cloud/test"


with open("test.pkl", "rb") as fp:
    info = pickle.load(fp)
    x_test = data_flower.all_h5(TESTSET)
enc_ac, ac = acae.network(return_enc=True)
ac.load_weights("weights_acae.h5")
enc_naive, naive = vnapf.network(return_enc=True)
naive.load_weights("weights.h5")
_, prob_ac = enc_ac.predict(x_test)
sort_ac = numpy.argsort(prob_ac, axis=-1)
_, prob_naive = enc_naive.predict(x_test)
sort_naive = numpy.argsort(prob_naive, axis=-1)


def extract(a_sort):
    Q = numpy.zeros([len(a_sort), 16, 3])
    for i in range(len(a_sort)):
        pcd, pr = x_test[i], a_sort[i]
        p_sort = numpy.reshape(pr[:, -2:], [-1])
        Q[i] = pcd[p_sort]
    return Q


kac = extract(sort_ac)
knaive = extract(sort_naive)
kusip = numpy.zeros([len(info), 16, 3])
for i in range(len(info)):
    kp, sigma = info[i]
    kp = kp[numpy.argsort(sigma)]
    kusip[i] = kp[:16]
print(kac.shape)
print(knaive.shape)
print(kusip.shape)
numpy.save("kac.npy", kac)
numpy.save("knaive.npy", knaive)
numpy.save("kusip.npy", kusip)
numpy.save("x_test.npy", x_test)
