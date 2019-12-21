# -*- coding: utf-8 -*-
"""
These are the solvers for the SVM

"""
from random import randint
import numpy as np
from cvxopt import matrix, spmatrix
from cvxopt import solvers

from sklearn.svm import SVC


""""""""""""""""""" Quadratic Program """""""""""""""""""""

class QP():
    def __init__(self, Kernel, slack=1, tol=1e-8):
        self.tol = tol
        self.K = Kernel.get_kernel()
        self.C = slack
        self.iter_count = None
        self.solution = None
        solvers.options['show_progress'] = False
        solvers.options['feastol'] = 1e-20
        self.status = None

    def run(self, X, y):
        H = _calculate_H(X, y, self.K)
        N = len(y)
        P = H # matrix(H)
        q = matrix(-1*np.ones(N))
        A = matrix(y, (1, N), 'd')
        b = matrix(0.0)
        G, h = _get_G_and_h(self.C, N)
        solution = solvers.qp(P, q, G, h, A, b)
        self.iter_count = solution['iterations']
        self.solution = np.ravel(solution['x'])
        self.status = solution['status']
        return self.solution


def _get_G_and_h(P, N):
    G_std = matrix(np.diag(np.ones(N) * -1))
    h_std = matrix(np.zeros(N))
    # a_i \leq c
    G_slack = matrix(np.diag(np.ones(N)))
    h_slack = matrix(np.ones(N) * P)
    # Put them together
    G = matrix(np.vstack((G_std, G_slack)))
    h = matrix(np.vstack((h_std, h_slack)))
    # Return the result
    return G, h


def _calculate_H(X, y, Kernel):
    N = len(X)
    H = spmatrix(0, (), (), (N, N))
    for i in range(N):
        for j in range(N):
            H[i, j] = y[i]*y[j]*Kernel(X[i], X[j])
    return H

""""""""""""""""""" SMO """""""""""""""""""""

class SMO():
    def __init__(self, Kernel, slack=1, tol=1e-8, max_passes=3e3):
        self.tol = tol
        self.K = Kernel.get_kernel()
        self.C = slack
        self.max_passes = max_passes

    def run(self, X, y):
        alphas = _SMO_Algorithm(self, self.C, self.tol, self.max_passes, X, y)
        return alphas
    


def _SMO_Algorithm(self, C, tol, max_passes, X, y):
    N = len(y)
    # Init alphas and b with 0
    alphas = np.zeros((N, ))
    b = 0
    passes = 0
    # Main loop
    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(N):
            Ei = F(self, alphas, y, X, X[i], b) - y[i]
            if (y[i]*Ei < -tol and alphas[i] < C) or (y[i]*Ei > tol and alphas[i] > 0):
                j = select_random(N, i)
                Ej = F(self, alphas, y, X, X[j], b) - y[j]
                old_ai = alphas[i]
                old_aj = alphas[j]
                L, H = compute_bounds(y[i], y[j], old_ai, old_aj, C)
                if L == H:
                    # print('L is H')
                    continue
                eta = compute_eta(self, X[i], X[j])
                if eta >= 0:
                    # print('Eta is 0')
                    continue
                alphas[j] = compute_new_alpha_j_(old_aj, y[j], Ei, Ej, eta, H, L)
                if abs(alphas[j] - old_aj) < 1e-5:
                    # print('No chanege in aj')
                    continue
                alphas[i] = compute_new_alpha_i_(old_ai, y[i], y[j], old_aj, alphas[j])
                b = compute_b_(self, b, Ei, Ej, y[i], y[j], alphas[i], alphas[j], old_ai, old_aj, X[i], X[j], C)
                num_changed_alphas += 1
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0
    return alphas


def F(self, alphas, yVect, XVect, X_SMO, b):
    return np.sum([alpha*y*self.K(xi, X_SMO) \
        for alpha, y, xi in zip(alphas, yVect, XVect)]) + b


def select_random(N, i):
    j = i
    while j == i:
        j = randint(0, N-1)
    return j

def compute_bounds(yi, yj, ai, aj, C):
    if yi != yj:
        L = max(0, aj - ai)
        H = min(C, C + aj - ai)
    else:
        L = max(0, ai + aj - C)
        H = min(C, ai + aj)
    return L, H

def compute_eta(self, xi, xj):
    return 2*self.K(xi, xj) - self.K(xi, xi) - self.K(xj, xj)

def compute_new_alpha_j_(old_aj, y, Ei, Ej, eta, H, L):
    aj = old_aj - y*(Ei-Ej)/eta
    if aj > H:
        return H
    elif aj < L:
        return L
    else:
        return aj

def compute_new_alpha_i_(old_ai, yi, yj, old_aj, new_aj):
    return old_ai + yi*yj*(old_aj - new_aj)


def compute_b_(self, b, Ei, Ej, yi, yj, ai, aj, old_ai, old_aj, Xi, Xj, C):
    b1 = b - Ei - yi*(ai - old_ai)*self.K(Xi, Xi) - yj*(aj - old_aj)*self.K(Xi, Xj)
    b2 = b - Ej - yi*(ai - old_ai)*self.K(Xi, Xj) - yj*(aj - old_aj)*self.K(Xj, Xj)
    if ai > 0 and ai < C:
        return b1
    elif aj > 0 and aj < C:
        return b2
    else:
        return (b1+b2)/2


""" ----------------- Sci-Kit Learn ----------------------"""

class SKL():
    def __init__(self, Kernel, slack=1, tol=1e-8):
        self.tol = tol
        self.K = Kernel
        self.C = slack
        self.sub_svm = SVC(slack, Kernel.get_type(), gamma='auto')
    
    def run(self, X, y):
        self.sub_svm.fit(X, y)
        return self.sub_svm.dual_coef_[0]


