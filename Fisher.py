import numpy as np
import pandas as pd
import random
import sys,itertools
import matplotlib.pyplot as plt
import scipy.stats

import copy

from scipy.stats import multivariate_normal

from mpl_toolkits.mplot3d import Axes3D

# joypy
import joypy
from matplotlib import cm
import cmocean as co


class Fisher:

    def __init__(self, feature_dim, binary_class = True, num_dims = 1):
        self.data_dict = {}
        self.class_means = {}
        self.feature_dim = feature_dim
        self.binary_class = binary_class
        self.num_dims = num_dims


    '''
    Fit method with helper functions
    '''
    def fit(self, x_train, y_train):
        # x_train, y_train numpy arrays
        self.total_samples = np.shape(x_train)[0]

        unique_labels = list(np.unique(y_train))
        for ul in unique_labels:
            idx_ul = np.where(y_train == ul)
            x_ul = x_train[idx_ul]
            self.data_dict[ul] = x_ul

        self._calculate_means()
        self._calculate_overall_mean(x_train)
        self._calculate_Sw()
        self._calculate_Sb()
        v1 = self._solve_eig()
        if self.binary_class:
            v2 = self._solve_inv()
        self._gaussian_modelling()

        return self.w

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

    def _gaussian_modelling(self):
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
        y_pred = self._calculate_score_gaussian(x_test)
        return y_pred   

    def _pdf(self, point, mean, cov):
        return multivariate_normal.pdf(point, mean=mean, cov=cov)

    def _pdf2(self, point, mean, cov):
        #hardcoded pdf
        cons = (1./((2*np.pi)**(len(point)/2.))*np.linalg.det(cov)**(-0.5))
        return cons*np.exp(-np.dot(np.dot((point-mean),np.linalg.inv(cov)),(point-mean).T)/2.)

    def _calculate_score_gaussian(self, x_test):
        classes = sorted(list(self.data_dict.keys()))
        # projection of test samples onto reduced dim
        proj = np.dot(self.w, x_test.T).T

        # likelihoods
        Y_pred = np.array([[self.priors[c] * self._pdf(x, self.gaussian_means[c], self.gaussian_covs[c]) for c in classes] for x in proj])
        y_pred = np.argmax(Y_pred, axis=1)

        return y_pred
        

    '''
    test the fit method of computing means, Sw, Sb
    '''
    def test_1(self, x_train, y_train):
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
    def test_2(self, x_train, y_train):
        self.fit(x_train, y_train)

        self._gaussian_modelling()

        return self.priors, self.gaussian_means, self.gaussian_covs

    def plot_joypy_1d(self, x_test, y_test):
        
        if self.num_dims != 1:
            print('Please fit the classifier to 1 dim reduced space in order to plot')
        else:
            classes = list(self.class_means.keys())
            colors = cm.rainbow(np.linspace(0, 1, len(classes)))
            plotlabels = {classes[c] : colors[c] for c in range(len(classes))}

            x_proj = np.dot(self.w, x_test.T).T

            result_df = pd.DataFrame(list(zip(np.squeeze(x_proj.real), y_test)), columns=['x_proj', 'label'])

            print(result_df.head())

            plt.figure(figsize=(6, 6))

            fig, axes = joypy.joyplot(result_df, column='x_proj', by='label', colormap=co.cm.thermal, grid=True, xlabelsize=12, ylabelsize=12)
            plt.title('Histograms from projection onto 1d space', fontsize=16, fontweight = 'bold')
            plt.xlabel('LDA axis', fontsize = 14)
            
            #plt.legend()
            plt.show()


    def plot_hist_1d(self, x_test, y_test, bins = 30):
        if self.num_dims != 1:
            print('Please fit the classifier to 1 dim reduced space in order to plot')
        else:
            classes = list(self.class_means.keys())
            colors = cm.rainbow(np.linspace(0, 1, len(classes)))
            plotlabels = {classes[c] : colors[c] for c in range(len(classes))}

            x_proj = np.dot(self.w, x_test.T).T

            fig, ax = plt.subplots(figsize=(8, 8))

            for c in classes:
                idx = np.where(y_test == c)
                x_proj_c = x_proj[idx]
                #ax.scatter(x_proj_c, np.random.normal(loc=0, scale=0.2, size=x_proj_c.shape[0]), color = plotlabels[c], label=c)
                ax.hist(x_proj_c, bins, normed = 1, facecolor = plotlabels[c], label=c)
            
            plt.legend()
            plt.show()


    def plot_proj_1d(self, x_test, y_test):
        if self.num_dims != 1:
            print('Please fit the classifier to 1 dim reduced space in order to plot')
        else:
            classes = list(self.class_means.keys())
            colors = cm.rainbow(np.linspace(0, 1, len(classes)))
            plotlabels = {classes[c] : colors[c] for c in range(len(classes))}

            x_proj = np.dot(self.w, x_test.T).T

            fig, ax = plt.subplots(figsize=(8, 8))

            for c in classes:
                idx = np.where(y_test == c)
                x_proj_c = x_proj[idx]
                ax.scatter(x_proj_c, np.random.normal(loc=0, scale=0.2, size=x_proj_c.shape[0]), color = plotlabels[c], label=c)
            plt.xlim([-1, 1])
            plt.show()

    def plot_proj_2d(self, x_test, y_test):

        if self.num_dims != 2:
            print('Please fit the classifier to 2 dim reduced space in order to plot')
        else:
            classes = list(self.class_means.keys())
            colors = cm.rainbow(np.linspace(0, 1, len(classes)))
            plotlabels = {classes[c] : colors[c] for c in range(len(classes))}

            x_proj = np.dot(self.w, x_test.T).T

            fig, ax = plt.subplots(figsize=(8, 8))

            for c in classes:

                idx = np.where(y_test == c)
                x_proj_c = x_proj[idx]
                ax.scatter(x_proj_c[:,0], x_proj_c[:,1], color = plotlabels[c], label=c)
            plt.legend()
            plt.show()


    def plot_proj_3d(self, x_test, y_test):
        if self.num_dims != 3:
            print('Please fit the classifier to 2 dim reduced space in order to plot')
        else:
            classes = list(self.class_means.keys())
            colors = cm.rainbow(np.linspace(0, 1, len(classes)))
            plotlabels = {classes[c] : colors[c] for c in range(len(classes))}

            x_proj = np.dot(self.w, x_test.T).T

            fig = plt.figure(figsize = (8,8))
            ax = fig.add_subplot(111, projection='3d')

            for c in classes:
                idx = np.where(y_test == c)
                x_proj_c = x_proj[idx]
                ax.scatter([float(d) for d in x_proj_c[:,0]], [float(d) for d in x_proj_c[:,1]], [float(d) for d in x_proj_c[:,2]], color=plotlabels[c], label=c)
            plt.legend()
            plt.show()


    