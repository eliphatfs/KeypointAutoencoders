# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 22:59:57 2020

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
import random
try:
    import open3d
except ImportError:
    print("WARNING: Error importing open3d")
import os
import h5py


def load_h5(h5_filename, normalize=False, include_label=False):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]  # (n, 2048, 3)
    if normalize:
        nmean = numpy.mean(data, axis=1, keepdims=True)
        nstd = numpy.std(data, axis=1, keepdims=True)
        nstd = numpy.mean(nstd, axis=-1, keepdims=True)
        data = (data - nmean) / nstd
    if include_label:
        label = f['label'][:]
        print(numpy.max(label))
        return data, label
    return data


def all_h5(parent, normalize=False, include_label=False):
    lazy = map(lambda x: load_h5(x, normalize, include_label),
               walk_files(parent))
    if include_label:
        xy = tuple(lazy)
        x = [x for x, y in xy]
        y = [y for x, y in xy]
        return numpy.concatenate(x), numpy.concatenate(y)
    return numpy.concatenate(tuple(lazy))


def walk_files(path):
    for r, ds, fs in os.walk(path):
        for f in fs:
            yield os.path.join(r, f)


def last_dirname(file_path):
    return os.path.basename(os.path.dirname(file_path))


def dataset_split(path):
    flist = list(walk_files(path))
    tr = filter(lambda p: 'train' in last_dirname(p), flist)
    te = filter(lambda p: 'test' in last_dirname(p), flist)
    return list(tr), list(te)


def rand_choose_gen(files, point_count):
    f = random.choice(files)
    mesh = open3d.io.read_triangle_mesh(f.replace("\\", "/"))
    if len(mesh.triangles) == 0:
        return rand_choose_gen(files, point_count)
    pc = mesh.sample_points_uniformly(point_count)
    return numpy.asarray(pc.points)


def flow(path, split, batch, point_count=1000):
    train_files, test_files = dataset_split(path)
    pc = point_count
    if pc < 2:
        raise ValueError("point_count should be >= 2, got", pc)
    if split == 'train':
        flist = train_files
    elif split == 'test':
        flist = test_files
    else:
        raise ValueError("split should be `train` or `test`, got", split)
    while True:
        pclist = list(map(lambda x: rand_choose_gen(flist, pc), range(batch)))
        yield numpy.array(pclist), numpy.array(pclist)
