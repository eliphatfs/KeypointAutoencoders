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
import time
import numpy
import pptk
import os


x_test = numpy.load("x_test.npy")
kusip = numpy.load("kusip.npy")
kac = numpy.load("kac.npy")
knaive = numpy.load("knaive.npy")
CLOUD, USIP, NAIVE, AC = 0, 1, 2, 3
view = [CLOUD, NAIVE, USIP, AC]


for i in range(len(x_test)):
    idx = numpy.random.randint(0, len(x_test))
    dest = [x_test]
    clds = x_test[idx], kusip[idx], knaive[idx], kac[idx]
    scls = [0] * len(x_test[idx]), [1] * len(kusip[idx]), [2] * len(knaive[idx]), [3] * len(kac[idx])
    xyz = numpy.concatenate(tuple(x for j, x in enumerate(clds) if j in view))
    sc = numpy.concatenate(tuple(x for j, x in enumerate(scls) if j in view))
    v = pptk.viewer(xyz, sc)
    v.set(point_size=0.007, show_grid=False, show_axis=False)
    v.color_map([[1.0, 1.0, 1.0, 0.6], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 1, 1]])
    q = input("Enter Picture Save Filename (Empty to Skip Saving, Q to Quit): ")
    numpy.savetxt("ipc.txt", x_test[idx])
    numpy.savetxt("okp0.txt", kusip[idx])
    numpy.savetxt("okp1.txt", knaive[idx])
    numpy.savetxt("okp2.txt", kac[idx])
    if len(q) > 0:
        if q == 'Q':
            v.close()
            break
        if not q.endswith(".png"):
            q += ".png"
        p = os.path.join(os.path.dirname(__file__), q)
        v.capture(p)
        time.sleep(1.0)
    v.close()
