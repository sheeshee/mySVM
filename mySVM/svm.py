# -*- coding: utf-8 -*-
"""

This file contains the class that will be the interface between the program
and the SVM and the user

"""
from time import clock
import numpy as np
import pandas as pd
from mySVM.solvers import SMO, QP

""" -------------- Classifier for multiclass identification ---------- """

class Classifier:
    def __init__(self, DataHandler, Solver, Kernel):
        self.Data = DataHandler
        self.Kernel = Kernel
        self.Solver = Solver
        self.iter_tracker = pd.DataFrame()
        self.SVMs = {}
        self.last_classification_results = None
        self.last_test_results = None
        clock()
    
    def train(self):
        """ Trains an SVM for each class to be used in WINNER-TAKE-ALL
        classification strategy """
        for dataclass in self.Data.get_class_names():
            print('Training for ', dataclass, '... ', end='')
            # train
            self.Data.set_class_of_interest(dataclass)
            self.SVMs[dataclass] = SVM(self.Data, self.Solver, self.Kernel)
            t = -clock()
            self.SVMs[dataclass].train()
            t += clock()
            self.iter_tracker.loc[dataclass, 'k'] = self.SVMs[dataclass].solver_iter_count
            self.iter_tracker.loc[dataclass, 'train time'] = t
            print('Complete!')

    def classify(self, uX):
        """ Accepts an unknown point uX and makes a prediction of what class
        it belongs to """
        results = pd.DataFrame(columns=['results'])
        for dataclass, SVM in self.SVMs.items():
            self.Data.set_class_of_interest(dataclass)
            _, score = SVM.predict(uX)
            results.loc[dataclass] = score
        self.last_classification_results = results
        winner = results.idxmax().at['results']
        return winner
    
    def test(self):
        """ Use the training set to evaluate the accuracy of the classifier """
        T_array = self.Data.getX('test')
        id_array = self.Data.getIDs('test')
        results = pd.DataFrame(columns=['prediction', 'actual', 'correct'])
        for ID, T in zip(id_array, T_array):
            prediction = self.classify(T)
            actual = self.Data.get_entry(ID)['label']
            results.loc[ID] = [prediction, actual, prediction==actual]
        accuracy = results['correct'].sum() / len(results)
        self.last_test_results = results
        return accuracy
    
    def minitest(self, N):
        """ Evaluates accuracy but for only N test elements """
        sample = self.Data.sample(N, 'test')
        X = sample.drop(['label', 'test'], axis=1)
        ids = sample.index.values
        results = pd.DataFrame(columns=['prediction', 'actual', 'correct'])
        for ID in ids:
            prediction = self.classify(X.loc[ID].values)
            actual = sample.at[ID, 'label']
            results.loc[ID] = [prediction, actual, prediction==actual]
        print("Score: %3.0f%%" % (results['correct'].sum()/len(results)*100))
        return results


class SVM:
    def __init__(self, DataHandler, Solver, Kernel, tol=1e-5):
        self.Data = DataHandler # Import Data Object
        self.Solver = Solver
        self.Kernel = Kernel.get_kernel()
        self.alphas = 0
        self.b = 0
        self.sv_cutoff = tol
        self.solver_iter_count = 0

    def loadData(self, DataHandler):
        """ Load the DataHandler object into the SVM """
        self.Data = DataHandler

    def setSolver(self, Solver):
    	""" Sets the internal solver to be used in training """
    	self.Solver = Solver

    def train(self):#, x0, mu0, tol, mu_mod, maxIter, C=1.0):
        """ Once the dataset has been loaded, train the SVM """
        X = self.Data.getX('train')
        y = self.Data.gety('train')
        # Find alphas
        alphas = self.Solver.run(X, y)
        if len(alphas) == len(y):
            alphas = y*alphas
        # Determine SVs
        svis = alphas[abs(alphas) > self.sv_cutoff]
        svis = svis[abs(svis) < (1 - self.sv_cutoff)]
        print("The support vector coefficients are:", sorted(svis))
        self.svis = svis
        # Calc b
        if len(self.svis) > 0:
            b = _calc_b(self, alphas, self.svis, X, y)
        else:
            b = 0
        self.alphas = alphas
        self.b = b

    def predict(self, X):
        """ Returns the predicted class of X """
        trainset_X = self.Data.getX('train')
        trainset_y = self.Data.gety('train')
        y = np.sum([alpha*y*self.Kernel(xi, X)
                    for alpha, y, xi in zip(self.alphas, trainset_y, trainset_X)])
        return np.sign(y), y

    def test(self):
        """ Returns the accuracy score of the SVM """
        X = self.Data.getX('test')
        y = self.Data.gety('test')
        correct = 0
        for yi, xi in zip(y, X):
            p, _ = self.predict(xi)
            if yi*p > 0:
                correct += 1
        return correct/self.Data.get_sample_count('test')
    
    def get_normal(self):
        """ Returns the normal of the dividing hyperplane """
        X = self.Data.getX('train')
        y = self.Data.gety('train')
        vals = [alphai*y[i]*X[i] for i, alphai in enumerate(self.alphas) ]
        w = np.sum(vals, axis=0)
        return w/np.linalg.norm(w)
    
    def get_training_time(self, N, iters=1):
        """ Returns how long it takes to train the SVM with a sample of N
        datapoints """
        frac = 1 - N/self.Data.get_sample_count('all')
        measurements = []
        for _ in range(iters):
            self.Data.reset_test_reserve(frac)
            _, t = _time_this(self.train)
            measurements.append(t)
        return sum(measurements)/len(measurements)


def _calculate_H(self):
    N = len(self.X)
    H = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            H[i, j] = self.y[i]*self.y[j]*self.Kernel(self.X[i], self.X[j])
    return H


def _calc_b(svm, alphas, svis, X, y):
    supersum = 0
    for i, _ in enumerate(svis):
        subsum = 0
        for j, _ in enumerate(svis):
            subsum += alphas[j]*y[j]*svm.Kernel(X[j], X[i])
        supersum += y[i] - subsum
    b = supersum/len(svis)
    return b


def _time_this(func):
    """ Times the execution time of function/method that does not
    take any input variables. Returns a tuple of the method result
    and the measured time"""
    t = -clock()
    result = func()
    t += clock()
    return result, t