# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 17:26:54 2020

@author: user

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
import data_flower
import visualizations
import pickle
import random
import acae


TESTSET = r"./point_cloud/test"


with open("test.pkl", "rb") as fp:
    info = pickle.load(fp)
    x_test = data_flower.all_h5(TESTSET)
enc, model = acae.network(return_enc=True)
model.load_weights("weights_acae.h5")
for i in range(len(info)):
    (kp, sigma), pcd = random.choice(list(zip(info, x_test)))
    Mx = enc.predict(numpy.array([pcd]))[1][0]
    MpSort = numpy.argsort(Mx, axis=-1)
    Mp = numpy.reshape(MpSort[:, -2:], [-1])
    kp = kp[numpy.argsort(sigma)]
    visualizations.color3(pcd, kp[:16], pcd[Mp])
