# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 21:32:06 2020

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
import pptk
import acae_kis4


TESTSET = "./point_cloud/test"
x_test = numpy.load("x_test_kis4.npy")
enc, model = acae_kis4.network(return_enc=True)
model.load_weights("weights_acae_k4.h5")
for i in range(len(x_test)):
    _, prob_t = enc.predict(numpy.array([x_test[i]]))
    for j in range(4):
        v = pptk.viewer(x_test[i], prob_t[0][j])
        v.set(point_size=0.008, show_grid=False, show_axis=False)
        v.color_map('cool')
        v.wait()
        v.close()
