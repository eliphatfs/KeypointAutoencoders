# -*- coding: utf-8 -*-
"""
Created on Thu May 21 10:05:45 2020

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
try:
    import open3d
    import numpy
except ImportError:
    print("WARNING: Error importing open3d")


def color3(p1, p2, p3):
    c = numpy.zeros_like(p1)
    c2 = numpy.zeros_like(p2)
    c3 = numpy.zeros_like(p2)
    c2[:, 0] = 1
    c3[:, 2] = 1
    cs = open3d.utility.Vector3dVector(numpy.concatenate((c2, c3, c)))
    pts = open3d.utility.Vector3dVector(numpy.concatenate((p2, p3, p1)))
    cloud = open3d.geometry.PointCloud()
    cloud.points = pts
    cloud.colors = cs
    # p1.paint_uniform_color(numpy.array([0, 0, 0]))
    # p2.paint_uniform_color(numpy.array([1, 0, 0]))
    # open3d.io.write_point_cloud()
    open3d.visualization.draw_geometries([cloud])


def merge_pcd(p1, p2):
    c = numpy.zeros_like(p1)
    c2 = numpy.zeros_like(p2)
    c2[:, 0] = 1
    cs = open3d.utility.Vector3dVector(numpy.concatenate((c, c2)))
    pts = open3d.utility.Vector3dVector(numpy.concatenate((p1, p2)))
    cloud = open3d.geometry.PointCloud()
    cloud.points = pts
    cloud.colors = cs
    # p1.paint_uniform_color(numpy.array([0, 0, 0]))
    # p2.paint_uniform_color(numpy.array([1, 0, 0]))
    # open3d.io.write_point_cloud()
    open3d.visualization.draw_geometries([cloud])


def show_point_cloud_array(pcd):
    pts = open3d.utility.Vector3dVector(pcd)
    cloud = open3d.geometry.PointCloud()
    cloud.points = pts
    open3d.visualization.draw_geometries([cloud])
