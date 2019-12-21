# -*- coding: utf-8 -*-
"""

This file contains the class that will be the interface between the program
and the SVM and the user

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class DataHandler:
    """ A class to contain and give access to the dataset with some helpful
    functions """
    def __init__(self, filename, test_segment=0.2):
        """ Creates the class by importing the data from the csv file and marking
        randomly entries to be used for training and which should be used for
        testing. The proprotion to be reserved for testing is determined by the
        test_segment parameter """
        self.raw = pd.read_csv(filename)
        self.raw = _mark_test_reserve(self.raw, test_segment)
        self.class_of_interest=None

    def get_dataset(self, choice='all'):
        """
        Returns a subset of the data selected by the choice parameter.
        If choice is 'test' it returns the test data set. If choice is 'train'
        it returns the training data set. If choice is all, it returns the entire
        dataset
        """
        if choice == 'train':
            return self.raw.loc[self.raw['test'] == 0]
        elif choice == 'test':
            return self.raw.loc[self.raw['test'] == 1]
        elif choice == 'all':
            return self.raw
        else:
            raise Exception("Invalid parameter passed. Must be 'all', 'test' or 'train'. Instead" \
            + " received: " + choice)

    def get_sample_count(self, choice='all'):
        """
        Returns the length of the dataset. Specify in choice if to get length of
        training set, test set or entire set.
        """
        if choice == 'test':
            return len(self.get_dataset('test'))
        if choice == 'train':
            return len(self.get_dataset('train'))
        else:
            return len(self.get_dataset())

    def getIDs(self, choice='all'):
        full = self.get_dataset(choice)
        return full.index.values

    def set_class_of_interest(self, labelname):
        self.class_of_interest=labelname
    
    def sample(self, N, choice='all'):
        """ Returns a sample of the dataset (choice can be test, train or all) """
        data = self.get_dataset(choice)
        return data.sample(N)
    
    def reset_test_reserve(self, frac):
        """ Re-initialises which datapoints are reserved for testing """
        self.raw = _mark_test_reserve(self.raw, frac)

class DigitData(DataHandler):
    """ Child class for interacting with the digit data set """
    def view_entry(self, id):
        """ Plots the data for the point at line id as a 28x28 pixel square """
        entry = self.get_entry(id)
        plt.imshow(_get_pixel_array(entry))
        plt.show()

    def get_entry(self, id):
        """ Returns a single entry as a dictionary with the pixel data reformated into
        a 2D numpy array """
        entry = self.raw.loc[id]
        pixels = entry.drop(['label', 'test']).values
        return {'label': entry['label'], 'test': entry['test'], 'features': pixels}

    def getX(self, choice='all'):
        full = self.get_dataset(choice)
        cols = [np.array(full[col_name].values).reshape(len(full), 1)
            for col_name in self.get_feature_column_names()]
        return np.concatenate(cols, axis=1)

    def gety(self, choice='all'):
        full = self.get_dataset(choice)
        y = [1 if label == self.class_of_interest else -1 for label in full['label'].values]
        return np.array(y)

    def get_class_names(self):
        return [i for i in range(10)]

    def get_feature_column_names(self):
        return ['pixel' + str(i) for i in range(28*28)]



class IrisData(DataHandler):
    """ Child class for interacting with the iris data set """
    def get_entry(self, id):
        entry = self.raw.loc[id]
        data = _get_iris_data(entry)
        return {'label': entry['label'], 'test': entry['test'], 'features': data}

    def plot_2D(self, slope=None, intercept=None):
        data = self.get_dataset('all')
        x = data['sepal_length']
        y = data['sepal_width']
        colors = [val for val in map(_iris_color_lookup, data['label'].values)]
        plt.scatter(x, y, c=colors)
        if slope:
            xx = np.linspace(4.5, 6)
            yy = slope * xx + intercept
            plt.plot(xx, yy)
        plt.show()

    def getX(self, choice='all'):
        full = self.get_dataset(choice)
        cols = [np.array(full[col_name].values).reshape(len(full), 1)
            for col_name in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        return np.concatenate(cols, axis=1)

    def gety(self, choice='all'):
        full = self.get_dataset(choice)
        y = [1 if species == self.class_of_interest else -1 for species in full['label'].values]
        return np.array(y)

    def get_class_names(self):
        full = self.get_dataset()
        return full.label.unique()


def _mark_test_reserve(data, test_segment):
    """ Adds a column to the dataframe named 'test' and populates it randomly
    with either 1's or 0' with 1's identifying the datapoints reserved for
    testing - in the proportion designated by the test_segment parameter """
    data['test'] = 0 # Initialise the row with 0's
    sample_indexes = data.sample(frac=test_segment).index
    data.loc[sample_indexes, 'test'] = 1
    return data


def _get_pixel_array(entry):
    """ Returns a 2D, 28x28 matrix corresponding to the image """
    entry = entry.drop(['label', 'test']).values
    pixel_array = np.reshape(entry, (28, 28))
    return pixel_array


def _get_iris_data(entry):
    """ Returns the features of the particular entry """
    return entry.drop(['test', 'label']).values


def _iris_color_lookup(species):
    if species == 'setosa':
        return 'r'
    elif species == 'versicolor':
        return 'g'
    elif species == 'virginica':
        return 'b'
    else:
        return 'k'