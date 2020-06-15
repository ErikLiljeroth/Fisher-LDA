import numpy as np
import pandas as pd
import random
import sys,itertools
import matplotlib.pyplot as plt
import scipy.stats

import copy

from scipy.stats import multivariate_normal

# 3d scatter plot
from mpl_toolkits.mplot3d import Axes3D

# plot colors
from matplotlib import cm
import cmocean as co


class Fisher:

    def __init__(self, num_dims = 1):
        '''Creates an instance of a Fisher Linear Discriminant Analysis (FLDA) classifier.
        Author: Erik Liljeroth

        Args:
            num_dims [int, optional]: The number of dimensions to project the feature space on. Defaults to 1.
        '''        
        self.data_dict = {}
        self.class_means = {}
        self.num_dims = num_dims

    def fit(self, x_train, y_train):
        '''Fits the FLDA classifier to training data.

        Args:
            x_train [numpy.array]: training data with observations along axis 0 and features along axis 1, e.g. shape = (5, 3) contains 5 samples with 3 features each
            y_train [numpy.array]: training labels

        Raises:
            ValueError: Raises error if y_train not includes at least 2 different labels

        Returns:
            W [np.array]: returns the computed projection matrix such that y = Wx, where x is a column vector of features for one sample
        '''        
        
        self.total_samples = np.shape(x_train)[0]
        self.feature_dim = x_train.shape[1]

        unique_labels = list(np.unique(y_train))

        if len(unique_labels) < 2:
            raise ValueError('All samples belong to same class... check the inputs to fit (especially y_train)')

        elif len(unique_labels) == 2:
            self.binary_class = True
        else:
            self.binary_class = False

        for ul in unique_labels:
            idx_ul = np.where(y_train == ul)
            x_ul = x_train[idx_ul]
            self.data_dict[ul] = x_ul

        self._calculate_means()
        self._calculate_overall_mean(x_train)
        Sw = self._calculate_Sw()
        self._calculate_Sb()
        self._solve_eig()
        self._estimate_multivariate_gaussian_parameters()
        print('-------------------------------')
        print('Fit process summary:')
        print('-------------------------------')
        if np.linalg.matrix_rank(Sw) < Sw.shape[0]:
            print('The within-class covariance matrix does not have full rank')
            print(f'Sw shape:{Sw.shape}, Sw rank:{np.linalg.matrix_rank(Sw)}')
        else:
            print('The within-class covariance matrix has full rank, yippie!')
        print('-------------------------------')

        return self.w

                
    '''
    Private helper functions for the fit method
    '''
    def _calculate_means(self):
        for k in self.data_dict.keys():
            self.class_means[k] = np.mean(self.data_dict[k], axis=0)
        return copy.deepcopy(self.class_means)

    def _calculate_overall_mean(self, x_train):
        self.overall_mean = np.mean(x_train, axis=0)

    def _calculate_Sw(self):
        self.Sw = np.zeros((self.feature_dim, self.feature_dim))

        for key in self.data_dict.keys():
            Sk = np.zeros((self.feature_dim, self.feature_dim))

            for d in self.data_dict[key]:
                Sk += np.outer((d-self.class_means[key]), (d-self.class_means[key]))

            self.Sw += Sk

        return copy.deepcopy(self.Sw)

    def _calculate_Sb(self):
        self.Sb = np.zeros((self.feature_dim, self.feature_dim))

        if self.binary_class:
            m0 = self.class_means[0]
            m1 = self.class_means[1]
            self.Sb = np.outer((m1-m0), (m1-m0))
        else:
            for key in self.data_dict.keys():
                self.Sb += len(self.data_dict[key]) * np.outer((self.class_means[key]-self.overall_mean), (self.class_means[key]-self.overall_mean))

        return copy.deepcopy(self.Sb)

    def _solve_eig(self):
        #mSw must be invertible
        mat = np.dot(np.linalg.pinv(self.Sw), self.Sb)
        e, v = np.linalg.eig(mat)
        eiglist = [(e[i], v[:,i]) for i in range(len(e))]
        # sort eigenvals in decreasing order
        eiglist = sorted(eiglist, key = lambda x : x[0], reverse=True)

        w = np.array([eiglist[i][1] for i in range(self.num_dims)])
        self.w = w

        return eiglist

    def _solve_inv(self):
        # Sw must be invertible
        if self.binary_class:
            Swi = np.linalg.inv(self.Sw)
            v = np.matmul(Swi, (self.class_means[0]-self.class_means[1]))
            return v
        else:
            print('Since the problem is not binary, this simple solution cannot be computed. Use the eigenvalue problem instead.')
            return None

    def _estimate_multivariate_gaussian_parameters(self):
        # prior probability
        self.priors = {}
        self.gaussian_means = {}
        self.gaussian_covs = {}

        for c in self.class_means.keys():
            inputs = self.data_dict[c]
            proj = np.dot(self.w, inputs.T).T
            self.priors[c] = inputs.shape[0]/self.total_samples
            self.gaussian_means[c] = np.mean(proj, axis=0)
            self.gaussian_covs[c] = np.cov(proj, rowvar=False)


    def predict(self, x_test):
        '''predicts labels using the fitted classifier

        Args:
            x_test [np.array]: observations along axis 0, features along axis 1

        Returns:
            y_pred [np.array]: 1D np.array of predicted labels
        '''        
        y_pred = self._calculate_score_gaussian(x_test)
        return y_pred   

    '''
    Private helper functions for the predict method
    '''
    def _pdf(self, point, mean, cov):
        return multivariate_normal.pdf(point, mean=mean, cov=cov)

    def _calculate_score_gaussian(self, x_test):
        classes = sorted(list(self.data_dict.keys()))
        # projection of test samples onto reduced dims
        proj = np.dot(self.w, x_test.T).T

        # likelihoods for different classes
        Y_pred = np.array([[self.priors[c] * self._pdf(x, self.gaussian_means[c], self.gaussian_covs[c]) for c in classes] for x in proj])
        y_pred = np.argmax(Y_pred, axis=1)

        return y_pred
        

    '''
    Plotting methods
    '''
    def plot_joypy_1d(self, x_test, y_test):
        '''Makes a joypy-plot of the projected data for 1D. Note: requires the joypy plotting library

        Args:
            x_train [numpy.array]: test data with observations along axis 0 and features along axis 1, e.g. shape = (5, 3) contains 5 samples with 3 features each
            y_train [numpy.array]: test labels
        '''        
        # joypy plots
        import joypy
        
        if self.num_dims != 1:
            print('Please fit the classifier to 1 dim reduced space in order to plot')
        else:

            x_proj = np.dot(self.w, x_test.T).T

            result_df = pd.DataFrame(list(zip(np.squeeze(x_proj.real), y_test)), columns=['x_proj', 'label'])


            fig, axes = joypy.joyplot(result_df, column='x_proj', by='label', colormap=co.cm.thermal, grid=True, xlabelsize=12, ylabelsize=12)
            plt.title('Histograms from projection onto 1d space', fontsize=16, fontweight = 'bold')
            plt.xlabel('LDA axis', fontsize = 14)
            


    def plot_hist_1d(self, x_test, y_test, bins = 30):
        '''Makes a standard matplotlib histogram-plot of the projected data for 1D.

        Args:
            x_train [numpy.array]: test data with observations along axis 0 and features along axis 1, e.g. shape = (5, 3) contains 5 samples with 3 features each
            y_train [numpy.array]: test labels
        '''
        if self.num_dims != 1:
            print('Please fit the classifier to 1 dim reduced space in order to plot')
        else:
            classes = list(self.class_means.keys())
            colors = co.cm.thermal(np.linspace(0.1, 0.8, len(classes)))
            plot_colors = {classes[c] : colors[c] for c in range(len(classes))}

            x_proj = np.dot(self.w, x_test.T).T

            fig, ax = plt.subplots(figsize=(8, 8))

            for c in classes:
                idx = np.where(y_test == c)
                x_proj_c = x_proj[idx]
                ax.hist(x_proj_c, bins, normed = 1, facecolor = plot_colors[c], label=c)
            
            plt.legend()
            plt.show()


    def plot_proj_2d(self, x_test, y_test):
        '''Makes a 2D scatter plot of the argument data if the classifier has been fitted to project the feature space on 2D

        Args:
            x_train [numpy.array]: test data with observations along axis 0 and features along axis 1, e.g. shape = (5, 3) contains 5 samples with 3 features each
            y_train [numpy.array]: test labels
        '''

        if self.num_dims != 2:
            print('Please fit the classifier to 2 dim reduced space in order to plot')
        else:
            classes = list(self.class_means.keys())
            colors = co.cm.thermal(np.linspace(0.1, 0.8, len(classes)))
            plot_colors = {classes[c] : colors[c] for c in range(len(classes))}

            x_proj = np.dot(self.w, x_test.T).T

            fig, ax = plt.subplots(figsize=(8, 8))

            for c in classes:

                idx = np.where(y_test == c)
                x_proj_c = x_proj[idx]
                ax.scatter(x_proj_c[:,0], x_proj_c[:,1], color = plot_colors[c], label=c)
            plt.legend()
            plt.show()


    def plot_proj_3d(self, x_test, y_test):
        '''Makes a 3D scatter plot of the argument data if the classifier has been fitted to project the feature space on 3D

        Args:
            x_train [numpy.array]: test data with observations along axis 0 and features along axis 1, e.g. shape = (5, 3) contains 5 samples with 3 features each
            y_train [numpy.array]: test labels
        '''
        if self.num_dims != 3:
            print('Please fit the classifier to 3 dim reduced space in order to plot')
        else:
            classes = list(self.class_means.keys())
            colors = co.cm.thermal(np.linspace(0.1, 0.8, len(classes)))
            plot_colors = {classes[c] : colors[c] for c in range(len(classes))}

            x_proj = np.dot(self.w, x_test.T).T

            fig = plt.figure(figsize = (8,8))
            ax = fig.add_subplot(111, projection='3d')

            for c in classes:
                idx = np.where(y_test == c)
                x_proj_c = x_proj[idx]
                ax.scatter([float(d) for d in x_proj_c[:,0]], [float(d) for d in x_proj_c[:,1]], [float(d) for d in x_proj_c[:,2]], color=plot_colors[c], label=c)
            plt.legend()
            plt.show()


    '''
    test the fit method of computing means, Sw, Sb
    '''
    def _test_1(self, x_train, y_train):
        # x_train, y_train numpy arrays

        unique_labels = list(np.unique(y_train))
        for ul in unique_labels:

            idx_ul = np.where(y_train == ul)
            x_ul = x_train[idx_ul]
            self.data_dict[ul] = x_ul

        means = self._calculate_means()
        Sw = self._calculate_Sw()
        Sb = self._calculate_Sb()

        return means, Sw, Sb

    '''
    Test the estimation of gaussian pdf's in the projected space
    '''
    def _test_2(self, x_train, y_train):
        self.fit(x_train, y_train)

        self._gaussian_modelling()

        return self.priors, self.gaussian_means, self.gaussian_covs



    