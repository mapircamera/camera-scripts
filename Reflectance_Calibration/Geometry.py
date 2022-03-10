# -*- coding: utf-8 -*-
"""
@author: MAPIR, Inc
"""

import numpy as np

def distance(a, b):
    return np.sqrt(np.power((a[0] - b[0]), 2) + np.power((a[1] - b[1]), 2))

def slope(a, b):
    return (b[1] - a[1]) / (b[0] - a[0])
def hypotenuse(points):
    line1 = distance(points[0], points[1])
    line2 = distance(points[1], points[2])
    line3 = distance(points[2], points[0])

    return max([line1, line2, line3])