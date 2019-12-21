# -*- coding: utf-8 -*-
"""

This file contains the Kernel class. An object that
returns a kernel function

"""

import numpy as np

class Kernel():
    def __init__(self, choice, param1=None, param2=None):
        self.kernel = set_kernel_by_choice(choice, param1, param2)
        self.choice = choice
    
    def get_kernel(self):
        return self.kernel
    
    def get_type(self):
        typ = self.choice
        if typ == 'linear':
            return 'linear'
        if typ == 'polynomial':
            return 'poly'
        if typ == 'gaussian':
            return 'rbf'
        if typ == 'sigmoid':
            return 'sigmoid'
        else:
            return 'other'


def set_kernel_by_choice(choice, param1, param2):
    if choice == 'linear':
        return lambda x1, x2: np.inner(x1, x2)
    elif choice == 'sigmoid':
        return lambda x1, x2: np.tanh(param1 * np.dot(x1, x2) + param2)
    elif choice == 'gaussian':
        return lambda x1, x2: np.exp(-1*param1*np.linalg.norm(np.subtract(x1, x2)))
    elif choice == 'polynomial':
        return lambda x1, x2: (param1 + np.inner(x1, x2)) ** param2
