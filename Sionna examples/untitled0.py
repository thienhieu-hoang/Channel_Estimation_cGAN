# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:09:46 2024

@author: AT30890
"""

import numpy as np

def some_sum_of(a,b):
    a1 = a*2
    b1 = b *3
    c = a1 + b1
    return c

a2 = np.array([3])
b2 = np.array([5])
somee = some_sum_of(a2, b2)
print(somee)