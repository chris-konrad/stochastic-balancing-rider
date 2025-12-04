# -*- coding: utf-8 -*-
"""
Model pole distributions representing behavioral parameters and test the distribution of their trajectory predictions.

@author: Christoph M. Konrad
"""

import os
# Prevent memory leakage of KNN on Windows
if os.name == 'nt':
    os.environ["OMP_NUM_THREADS"] = "1"

import re
import yaml
import warnings

import numpy as np
import pandas as pd
import pickle as pkl

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredText
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

from datetime import datetime

from sklearn.mixture import GaussianMixture as SklearnGaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from sklearn.preprocessing import StandardScaler, PowerTransformer, FunctionTransformer
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression

from scipy.stats import multivariate_normal, gaussian_kde, ks_2samp

from rcid.simulation import FixedInputZigZagTest, FixedSpeedStepResponses
from rcid.simulation import FixedSpeedBalancingRiderBicycle, FixedSpeedPlanarPointBicycle
from rcid.utils import read_yaml
from rcid.path_manager import PathManager
from rcid.data_processing import RCIDDataManager

from pypaperutils.design import TUDcolors

tudcolors = TUDcolors()
cmap = tudcolors.colormap()

#global constants
T_S = 0.01

def get_outliers_all_models(paths, models):
    """ Combine outliers across models. 
    """
    
    outlier_cols = []
    for i, m in enumerate(models):
        filepath_poles = paths.getfilepath_pm_sortedpoles(m)

        df_i = pd.read_csv(filepath_poles, sep=";")[['sample_id', 'outliers']]
        col = f'outliers_{m}'
        outlier_cols.append(col)

        if i == 0:
            df = df_i.rename(columns={'outliers': col})
        else:
            df = df.merge(df_i, on='sample_id')
            df.rename(columns={'outliers': col}, inplace=True)
            
    df['outliers'] = df[outlier_cols].any(axis=1)

    print(f"Number of samples: {df.shape[0]-df['outliers'].sum()} (ignoring {df['outliers'].sum()} outliers across {models})")

    return df

def polefeaturetable_to_polearray(polefeature_table, features='ImRe'):
    """ Convert pole features in a pandas table into an array of complex-valued poles
    """

    poles = []

    if features == 'ImRe':
        for i in range(10):
            key_real = f"p{i}_real"
            key_imag = f"p{i}_imag"

            p_i = np.zeros(polefeature_table.shape[0], dtype=complex)

            if key_real in polefeature_table:
                p_i += polefeature_table[key_real].to_numpy().flatten()
            if key_imag in polefeature_table:
                p_i += 1j * polefeature_table[key_imag].to_numpy().flatten()
            
            if (key_imag not in polefeature_table) and (key_real not in polefeature_table):
                break

            poles.append(p_i)

            if np.any(np.imag(p_i) != 0.0):
                poles.append(np.conjugate(p_i))

    elif features == 'AngMag':
        for i in range(10):
            key_real = f"p{i}_real"
            key_ang = f"p{i}_ang"
            key_mag = f"p{i}_mag"

            p_i = np.zeros(polefeature_table.shape[0], dtype=complex)

            if key_real in polefeature_table:
                p_i += polefeature_table[key_real].to_numpy().flatten()
            elif key_ang in polefeature_table and key_mag in polefeature_table:
                p_i += polefeature_table[key_mag] * (np.cos(polefeature_table[key_ang]) + 1j * np.sin(polefeature_table[key_ang]))
            else:
                break

            poles.append(p_i)

            if np.any(np.imag(p_i) != 0.0):
                poles.append(np.conjugate(p_i))
            
    poles = np.array(poles).T

    return poles


def score_gmm(gmm, X):
    """ Compute the multimetric score of a gaussian mixture model.
    """
    if type(gmm) != GaussianMixture:
        raise ValueError(f"'gmm' must be sklearn.mixture.GaussianMixture. Instead it was {type(gmm)}")

    score = {'BIC': gmm.bic(X),
            'AIC': gmm.aic(X),
            'NLL': -gmm.score(X)}
    return score
    

def score_conditional_gmm(gmm, X):
    """ Compute the multimetric score of a conditional gaussian mixture model.
    """
    
    if type(gmm) != ConditionalGaussianMixture:
        raise ValueError(f"'gmm' must be ConditionalGaussianMixture. Instead it was {type(gmm)}")

    scores = []

    feature_index_rest = [n for n in range(X.shape[1]) if n != gmm.feature_index_given]

    for i in range(X.shape[0]):
            
        X_given = X[i, gmm.feature_index_given]
        X_i = X[i, feature_index_rest].reshape(1, len(feature_index_rest))

        gmm_cond = gmm._get_conditional_gmm(X_given)

        scores.append([gmm_cond.bic(X_i), gmm_cond.aic(X_i), -gmm_cond.score(X_i)])

    scores = np.array(scores)
    scores = np.mean(scores, axis=0)

    return {'BIC': scores[0],
            'AIC': scores[1],
            'NLL': scores[2]}



class GaussianMixture(SklearnGaussianMixture):
    """ A class extending sklearn's Gaussian Mixture with functionality to create objects from known parameters,
    and custom pdf evaluation. Additionally enables to scale the variance of all components.
    """

    def __init__(self, n_components=1, n_init=100, covariance_type='full', variance_scale=1.0, **kwargs):
        """ Create a GaussianMixture object.

        Parameters
        ----------

        n_components : int, optional
            Number of Gaussian components. Default is 1.
        n_init : int, optional
            Number of initializations for fitting. Defualt is 100.
        covariance_type : str, optional
            The covariance type. Default is "full".
        variance_scale : float, optional
            Variance scale factor applied to all components. Default is 1.0.
        kwargs : dict
            All other keyword arguments from Sklearn's GaussianMixture.
        """
        self.variance_scale=variance_scale
        super().__init__(n_components=n_components, covariance_type=covariance_type, n_init=n_init, **kwargs)


    def from_parameters(means, covariances, weights, **kwargs):
        """ Create a multivariate Gaussian Mixture model from known / converged parameters.

        Parameters
        ----------
        means : array-like
            Array of means. Must be shaped (n_features, n_components).
        covariances : array-like
            Array of covariances. Must be shaped (n_components, n_features, n_features).
        weights : array-like
            Array of component weights. Must be size n_components. Weights must sum to 1.0
        kwargs : dict
            Any OTHER keyword arguments for sklearn.mixture.GaussianMixture

        Returns
        -------
        gmm : GaussianMixture
            A GaussianMixture object with the given parameters. 

        """

        means = np.array(means)
        covariances = np.array(covariances)

        n_features = means.shape[1]
        n_components = means.shape[0]

        if not np.all(covariances.shape == np.array((n_components, n_features, n_features))):
            msg = (f"n_features={n_features} and n_components={n_components} inferred from means.shape. "
                   f"'covariances' must be shaped [{n_components},{n_features},{n_features}]. Instead it "
                   f"was {covariances.shape}")
            raise ValueError(msg)
        
        weights = np.array(weights).flatten()
        if weights.size != n_components:
            msg = (f"n_components={n_components} inferred from means.shape. "
                   f"'weights' must be size {n_components}. Instead it "
                   f"was {weights.size}")
        if np.sum(weights) != 1.0:
            msg = (f"Weights do not sum to one!")
        
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', **kwargs)
        gmm.means_ = means
        gmm.covariances_ = covariances
        gmm.weights_ = weights
        gmm.precisions_cholesky_ = _compute_precision_cholesky(covariances, 'full')
        gmm.converged_ = True
        gmm.n_iter_ = 1

        return gmm
    
    def fit(self, X):
        """ Fit the Gaussian Mixture model to the given Data.

        Parameters
        ----------
        X : array-like
            Data matrix.
        """

        super().fit(X)

        if self.variance_scale != 1.0:
            cov = self.get_full_covariancematrix()
            S = np.eye(cov.shape[1]) * np.sqrt(self.variance_scale)
            for i in range(cov.shape[0]):
                cov[i,:,:] = S @ cov[i,:,:] @ S.T

            self.covariances_ = cov
            self.covariance_type = 'full'
            self.precisions_cholesky_ = _compute_precision_cholesky(cov, 'full')

        return self

    def get_full_covariancematrix(self):
        """ Return the full covariance matrix of the fitted Gaussian Mixture model, 
        even if the covariance type is tied/diag or spherical.

        Returns
        -------
        covariances : array like
            Covariance matrices shaped [n_features, n_features, n-components]
        """
        n_features = self.means_.shape[1]

        if self.covariance_type == 'full':
            return self.covariances_
        elif self.covariance_type == "tied":
            return np.tile(self.covariances_[np.newaxis, :, :], (self.n_components,1,1))
        elif self.covariance_type == "diag":
            return np.array([np.diag(self.covariances_[k]) for k in range(self.n_components)])
        elif self.covariance_type == 'spherical':
            return np.array([np.eye(n_features) * self.covariances_[k] for k in range(self.n_components)])
        raise RuntimeError(f"Illegal covariance type {self.covariance_type}!")
    
    def eval_1d_marginal_pdf_samples(self, samples, idx_x):
        """ Evaluate the marignal pdf of a selected feature x at sample locations.

        Parameters
        ----------
        samples : array_like
            Sample values of feature x to evaluate the marinal pdf at.
        idx_x : int
            The index of feature x. 

        Returns
        -------
        samples : array_like
            Samples of feature x (same as input)
        densities : array_like
            Marginal densities of feature x at the sample locations
        """

        # accumulate densities
        densities = np.zeros_like(samples)
        cov = self.get_full_covariancematrix()
        
        for k in range(self.n_components):
            mean_k = self.means_[k, idx_x]
            var_k = cov[k][idx_x, idx_x]

            densities_k = self.weights_[k] * multivariate_normal(mean=mean_k, cov=var_k).pdf(samples)
            densities += densities_k

        return samples.flatten(), densities.flatten()

    def eval_1d_marginal_pdf(self, xlim, idx_x, n_samples=200):
        """ Evaluate the marignal pdf of a selected feature x across a range.

        Parameters
        ----------
        xlim : tuple
            A tuple specifying the range to evaluate as [min, max].
        idx_x : int
            The index of feature x. 
        n_samples : int, optional
            Number of samples with the given limits. Default is 200.

        Returns
        -------
        samples : array_like
            Samples of feature x (same as input)
        densities : array_like
            Marginal densities of feature x at the sample locations
        """
        
        # grid
        locations = np.linspace(xlim[0], xlim[1], n_samples)

        return self.eval_1d_marginal_pdf_samples(locations, idx_x)

    def eval_2d_marginal_pdf(self, xlim, ylim, idx_x, idx_y, n_samples=200):
        """ Evaluate the 2d marignal pdf of a pair of features x and y across a range.

        Parameters
        ----------
        xlim : tuple
            A tuple specifying the range of feature x to evaluate as [min, max].
        ylim : tuple
            A tuple specifying the range of feature y to evaluate as [min, max].
        idx_x : int
            The index of feature x. 
        n_samples : int, optional
            Number of samples with the given limits. Default is 200.

        Returns
        -------
        samples : array_like
            Samples of feature x (same as input)
        densities : array_like
            Marginal densities of feature x at the sample locations
        """

        # grid
        x = np.linspace(xlim[0], xlim[1], n_samples)
        y = np.linspace(ylim[0], ylim[1], n_samples)
        X, Y = np.meshgrid(x, y)
        locations = np.dstack((X, Y))

        # accumulate densities
        densities = np.zeros_like(X)
        cov = self.get_full_covariancematrix()
        

        for k in range(self.n_components):
            mean_k = [self.means_[k, idx_x], self.means_[k, idx_y]]
                
            cov_k = cov[k]
            cov_k = cov_k[[idx_x, idx_y],:][:, [idx_x, idx_y]]

            densities_k = self.weights_[k] * multivariate_normal(mean=mean_k, cov=cov_k).pdf(locations)
            densities += densities_k

        return locations.reshape(-1,2), densities.flatten()


class ConditionalGaussianMixture(GaussianMixture):
    """ Describes a conditional multivariate GaussianMixture.
    """

    def __init__(self, feature_index_given=1, n_components=1, n_init=100, covariance_type='full', **kwargs):
        """ Create a ConditionalGaussianMixture object.

        Parameters
        ----------
        feature_index_given : int, optional
            The index of the conditional feature in the data matrix. 
        n_components : int, optional
            Number of Gaussian components. Default is 1.
        n_init : int, optional
            Number of initializations for fitting. Defualt is 100.
        covariance_type : str, optional
            The covariance type. Default is "full".
        variance_scale : float, optional
            Variance scale factor applied to all components. Default is 1.0.
        kwargs : dict
            All other keyword arguments from Sklearn's GaussianMixture and this modules' GaussianMixture.
        """

        super().__init__(n_components=n_components, covariance_type=covariance_type, n_init=n_init, **kwargs)

        self.feature_index_given = feature_index_given

        #if not self.covariance_type == 'full':
        #    raise ValueError("Covariance type must be 'full'!")

    
    def from_parameters(means, covariances, weights, feature_index_given, **kwargs):
        """ Create a multivariate Gaussian Mixture model from known / converged parameters.

        Parameters
        ----------
        means : array-like
            Array of means. Must be shaped (n_features, n_components).
        covariances : array-like
            Array of covariances. Must be shaped (n_components, n_features, n_features).
        weights : array-like
            Array of component weights. Must be size n_components. Weights must sum to 1.0
        feature_index_given : int
            Index of the given feature.
        kwargs : dict
            Any OTHER keyword arguments for sklearn.mixture.GaussianMixture

        Returns
        -------
        gmm : GaussianMixture
            A GaussianMixture object with the given parameters. 

        """

        means = np.array(means)
        covariances = np.array(covariances)

        n_features = means.shape[1]
        n_components = means.shape[0]

        if not np.all(covariances.shape == np.array((n_components, n_features, n_features))):
            msg = (f"n_features={n_features} and n_components={n_components} inferred from means.shape. "
                   f"'covariances' must be shaped [{n_components},{n_features},{n_features}]. Instead it "
                   f"was {covariances.shape}")
            raise ValueError(msg)
        
        weights = np.array(weights).flatten()
        if weights.size != n_components:
            msg = (f"n_components={n_components} inferred from means.shape. "
                   f"'weights' must be size {n_components}. Instead it "
                   f"was {weights.size}")
        if np.sum(weights) != 1.0:
            msg = (f"Weights do not sum to one!")
        
        gmm = ConditionalGaussianMixture(feature_index_given=feature_index_given, n_components=n_components, covariance_type='full', **kwargs)
        gmm.means_ = means
        gmm.covariances_ = covariances
        gmm.weights_ = weights
        gmm.precisions_cholesky_ = _compute_precision_cholesky(covariances, 'full')
        gmm.converged_ = True
        gmm.n_iter_ = 1

        return gmm


    def fit(self, X):
        """ Fit the conditional Gaussian Mixture model to the given data.
        Should include samples for the feature to be conditioned on. 

        Parameters
        ----------
        X : array-like
            Data matrix.
        """
        self.feature_indices_marginals = [i for i in np.arange(X.shape[1]) if i != self.feature_index_given]
        super().fit(X)
        return self


    def _get_conditional_gmm(self, x_given):
        """ Return a GaussianMixture object modeling the distribution conditioned on X_given"""

        cov = self.get_full_covariancematrix()
        mu = self.means_
        pi = self.weights_

        idx_given = np.array(self.feature_index_given)
        idx_cond = [n for n in range(self.means_[0].size) if n not in idx_given]

        n_features = self.means_.shape[1]
        n_given = idx_given.size
        n_cond = n_features - n_given

        x_given = np.reshape(x_given, (n_given, 1))
        
        # make masks to form the conditional covariance matrices
        # given: the given feature to be conditioned on
        # cond: the remaining conditional distribution
        mask_given = np.zeros((n_features, n_features), dtype=bool)
        mask_given[idx_given, :] = True
        mask_given[:, idx_given] = True

        mask_cond_cov = np.logical_not(mask_given)

        cov_cond = []
        mu_cond = []
        pi_cond = []

        for n in range(self.n_components):
            cov_n = cov[n,:,:]
            mu_n = mu[n,:]

            var_given_n = cov_n[idx_given, idx_given].reshape(n_given,n_given)
            cov_given_n = cov_n[idx_cond, idx_given].reshape(n_cond,n_given)
            mu_given_n = mu_n[idx_given].reshape(n_given, 1)

            mu_cond_n = mu_n[idx_cond].reshape(n_cond, 1) + (cov_given_n @ np.linalg.inv(var_given_n)) @ (x_given - mu_given_n)
            cov_cond_n = cov_n[mask_cond_cov].reshape(n_cond, n_cond) - (cov_given_n @ np.linalg.inv(var_given_n) @ cov_given_n.T)

            pi_cond_n = pi[n] * multivariate_normal.pdf(x_given, mu_given_n, var_given_n)

            cov_cond.append(cov_cond_n)
            mu_cond.append(mu_cond_n.flatten())
            pi_cond.append(pi_cond_n)

        pi_cond = np.array(pi_cond) / np.sum(pi_cond)
        if np.any(pi_cond==0.0):
            #prevent pi_cond from getting 0 to suppress warnings later on
            pi_cond[pi_cond == 0.0] = np.finfo(float).eps * self.n_components
            pi_cond = pi_cond/np.sum(pi_cond)
        cov_cond = np.array(cov_cond)

        gmm_cond = GaussianMixture.from_parameters(mu_cond, cov_cond, pi_cond, random_state=self.random_state)

        return gmm_cond 
    
    
    def sample(self, n_samples=1, X_given=[0.0]):
        """ Draw samples from the conditional distribution. Draws n_samples per given feature value X_given to be conditioned on.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to draw. Default is 1.
        X_given : list-like, optional
            List of n_given feature values to be conditioned on. Default is [0.0].
        
        Returns
        -------
        samples : np.ndarray
            Array of drawn samples shape (n_given, n_samples, n_features). If n_samples==1, the shape is (n_samples, n_features)
        """ 

        if not isinstance(X_given, (list, tuple, np.ndarray)):
            X_given = list(X_given)
        
        samples = []
        labels = []
        for x_given in X_given:
            gmm_cond = self._get_conditional_gmm(x_given)
            samples_i, labels_i = gmm_cond.sample(n_samples=n_samples)
            samples.append(samples_i)
            labels.append(labels_i)

        if len(X_given)>1:
            samples = np.array(samples)
        else:
            samples = samples_i

        labels = np.array(labels).flatten()

        return samples, labels
    

    def eval_conditional_marginal_pdf(self, ylim, x_given, idx_y, n_samples=200):
        """ Evaluate the marginal conditional pdf N(Y=y|X=x_given). 

        Parameters
        ----------

        ylim : list
            Range of y
        x_given : float
            The given value to be conditioned on
        idx_y : int
            The id of the requested marginal. May not be the ID of the conditional. 
        n_samples : int
            Number of samples for the marginal distribution between ylim[0] and ylim[1]
        """

        if idx_y == self.feature_index_given:
            raise ValueError("The requested marginal can't be the one that is conditoned on!")
        
        #convert to index of the conditional distribution
        idx_y = self.feature_indices_marginals.index(idx_y)
        
        #conditional gmm
        gmm_cond = self._get_conditional_gmm(x_given)

        #range
        y = np.linspace(ylim[0], ylim[1], n_samples)
        densities = np.zeros_like(y)
    
        #accumulate densities
        for k in range(self.n_components):

            mean_k = gmm_cond.means_[k][idx_y]
            cov_k = gmm_cond.covariances_[k][idx_y, idx_y]

            densities += gmm_cond.weights_[k] * multivariate_normal.pdf(y, mean=mean_k, cov=cov_k)
        
        return y, densities



class PoleSorter():

    # CLASS CONSTANTS
    SORTING_METHODS = ["magnitude", "argument", "frequency", "decay", "none"]
    REQUIRED_PATHS = ["filepath_identification_result", "dir_out"]
    SUBDIRS = {"output-dirname": "pole-modeling"}
    OUTPUT_FNAMES = {"standard-stepresponses": "standard-stepresponses",
                     "pole-sorting-result": "sorted-poles"}

    def __init__(self, paths, method_real="decay", method_complex="frequency", riderbike_model='BR0', threshold_obj=10e-4, gainlimits=[-250,250],
                 timedomain_outlier_zscore=3, timedomain_outlier_testperiod=[0.5,10], bikemodel='balancingrider', save=False):
        """ Create a pole sorter object.

        Sorts poles according to the selected sorting method, identifies outliers and converts poles to pole features.
        Poles are classified as outliers if:
            - they exceed an objective value threshold.
            - they hit the gain limits.
            - their step response is a timedomain outlier in the distribution of all step responses.

        Parameters
        ----------

        paths : dict
            Dictionary of input paths containing: "filepath_identification_result", "dir_out"
        method_real : str, optional
            Sorting method for the real part of the poles. Must be any of ["magnitude", "argument", "frequency", "decay", "none"].
            Default is "decay".
        method_complex : str, optional
            Sorting method for the complex part of the poles. Must be any of ["magnitude", "argument", "frequency", "decay", "none"].
            Default is "frequency".
        riderbike_model : str, optional
            Model id of the riderbikemodel to build the model on. Must be any of the riderbikemodels in the config. Default is
            "BR0". 
        threshold_obj : float, optional
            Identification objective value threshold to classify a set op poles as outliers. Default is 10e-4.
        gain_limits : list or dict, optional
            Identification gain limits for classifying identifications at the gain limits as outliers. Must be a list [min, max]
            that is valid for all gains or a dictionary of limits per gain {gain: [min, max]}. Default is [-250, 250].
        timedomain_outlier_zscore : int, optional
            Number of standard deviations to classify the step response of a set of poles as an outlier.
        timedomain_outlier_testperiod : list, optional
            Time period to evaluate the distribution of step responses generated by the pole sets for detecting outliers.
        bikemodel : str, optional
            The riderbike model type. E.g., 'balancingrider' or 'planarpoint'. The default is 'balancingrider'. 
        save : bool, optional
            Save the results to file. Default is True. 
        """

        #settings
        self.method_real = self._check_method(method_real)
        self.method_complex = self._check_method(method_complex)

        #data and outputr
        self.save = save
        self.paths = self._check_paths(paths)
        self.riderbike_model = riderbike_model

        self.identifications = pd.read_csv(self.paths["filepath_identification_result"], sep=";")

        #outlier detection
        self.timedomain_outlier_zscore = timedomain_outlier_zscore
        self.timedomain_outlier_testperiod = timedomain_outlier_testperiod
        self.bikemodel = bikemodel

        self.identification_outlier_thobj = threshold_obj
        self.identification_outlier_gainlimits = gainlimits

        
    def _check_method(self, method):

        if method not in self.SORTING_METHODS:
            raise ValueError(f"Sorting method must be any of {self.SORTING_METHODS}. Instead it was '{method}'.")

        return method
    

    def _check_paths(self, paths):
        """ Check that all required paths are supplied and exist. """

        for p in self.REQUIRED_PATHS:
            if p not in paths:
                raise ValueError(f"Path to {p} missing in 'paths'. 'paths' must have at least {self.REQUIRED_PATHS}.")
            if 'filepath' in p:
                if not os.path.isfile(paths[p]):
                    raise IOError(f"Can't find file '{p}' at {paths[p]}.")
            elif 'dir' in p:
                if not os.path.isdir(paths[p]):
                    raise IOError(f"Can't find directory '{p}' at {paths[p]}.")
        
        # if necessary, make output directory
        if not os.path.basename(os.path.normpath(paths['dir_out'])) == self.SUBDIRS["output-dirname"]:
            paths['dir_out'] = os.path.join(paths['dir_out'], self.SUBDIRS["output-dirname"])
        if not os.path.isdir(paths['dir_out']):
            os.makedirs(paths['dir_out'])

        return paths


    def _get_standardstepresponses(self):
        """ Return the standard step responses. 

        Try to load the date from file. If not available, simulate standard step responses and dump them for later. 
        """

        poles = self.identifications[np.logical_not(self.identifications['outliers'])]
        
        # step input data
        pole_array = polefeaturetable_to_polearray(poles)
        speeds = poles['v_mean'].to_list()
        files = poles['sample_id'].to_list()

        # file name of standard step responses
        filepath_steps = self.OUTPUT_FNAMES["standard-stepresponses"] + ".pkl"
        if self.riderbike_model is not None:
            filepath_steps = self.riderbike_model + "_" + filepath_steps
        filepath_steps = os.path.join(self.paths['dir_out'], filepath_steps)

        # step response object
        sim = FixedSpeedStepResponses(pole_array, speeds, T=self.timedomain_outlier_testperiod[1], bikemodel=self.bikemodel, psi_c_deg=45)
        
        if os.path.isfile(filepath_steps):
            
            with open(filepath_steps, 'rb') as f:
                dump = pkl.load(f)

            #check that poles and dumped stepresponses match
            test_speeds = dump['speeds'] == speeds
            test_files = dump['sample_id'] == files
            if dump['poles'].size != pole_array.size:
                test_poles = False
            else:
                test_poles = np.all(dump['poles'] == pole_array)
            if not (test_files and test_speeds and test_poles):
                raise ValueError(f"The recovered standard stepresponses at {filepath_steps} do not correspond to the current pole data!")
            
            trajs = dump['trajs']

        else:
            trajs = sim.simulate()

            if self.save:
                dump = dict(trajs=trajs, speeds=speeds, sample_id=files, poles=pole_array, bikemodel=self.bikemodel)
                with open(filepath_steps, 'wb') as f:
                    pkl.dump(dump, f)

        #test_loc_metrics = [f'LD-t{100*i}' for i in np.arange(5,50,2)]
        speed_ideal = 4
        x_ideal, y_ideal, psi_ideal = sim.get_ideal_xy_response(speed=speed_ideal)
        evaluator = SampleEvaluator(x_ideal[:-2], y_ideal[:-2], psi_ideal[:-2], sim.p_x_c[:-1], sim.p_y_c[:-1], "standard stepresponse", 0, 0, 0, 0, speed_ideal)

        #figX, axX = plt.subplots(1,1)
        LD = np.zeros((1, trajs[0].shape[1], len(trajs)))
        for i, trj in enumerate(trajs):
            xy = evaluator._rotate_sample(trj[0,:], trj[1,:])
            LD[0, :, i] = xy[1,:]
            #axX.plot(LD[0, :, i])
            evaluator.add_pred_sample(i, trj[0,:], trj[1,:], trj[2,:])

        #evaluator.eval_groundtruth_metrics()
        #evaluator.plot_histograms()

        return trajs, LD, pole_array, poles


    def _find_outliers_responsetime(self):
        """ Step response samples where a valid response time could not be identified.
        """

        responsetimes = pd.read_csv(self.paths["filepath_responsetime_identifications"], sep=",")

        outliers_rt = np.zeros(self.identifications.shape[0], dtype=bool)
        for _, row in responsetimes.iterrows():
            idx = np.argwhere(row['sample_id']==self.identifications['sample_id'][:-4]).flatten()
            outliers_rt[idx] = row['outlier_response_time']

        n_outliers_rt = np.sum(np.logical_and(outliers_rt, np.logical_not(self.identifications['outliers'])))
        print(f"    + N(response time): {n_outliers_rt} ({100*n_outliers_rt/self.identifications.shape[0]:.1f} %)")

        self.identifications['outliers_response_time'] = outliers_rt
        self.identifications['outliers'] = np.logical_or(outliers_rt, self.identifications['outliers'])
    
    
    def _find_outliers_objective(self):
        """ Identify identification results that exceed the objective value threshold self.identification_outlier_thobj
        """

        # bad objective value
        outliers_objective = self.identifications['objective'] > self.identification_outlier_thobj
        n_outliers_objective = np.sum(np.logical_and(outliers_objective, np.logical_not(self.identifications['outliers'])))
        print(f"    + N(objective > {self.identification_outlier_thobj}): {n_outliers_objective} ({100*n_outliers_objective/self.identifications.shape[0]:.1f} %)")

        self.identifications['outliers_objective'] = outliers_objective
        self.identifications['outliers'] = np.logical_or(outliers_objective, self.identifications['outliers'])

    
    def _find_outliers_gainlimit(self):
        """ Identify identification results that hit the gain limits. Only results at limits =/= 0 are returned. 
        """

        # at gain limits
        outliers_gains = np.zeros(self.identifications.shape[0], dtype=bool)
        for k in self.identification_outlier_gainlimits:
            lim = np.array(self.identification_outlier_gainlimits[k])
            if lim[0] == 0.0:
                lim[0] = -np.inf
            if lim[1] == 0.0:
                lim[1] = np.inf
            if k == 'k_psi':
                lim = np.array([0])
            else:
                continue
            lim = np.reshape(lim, (1,-1))
            outliers_k = np.sum(np.abs((lim - self.identifications[k].to_numpy()[:,np.newaxis])) < 0.01, axis=1).astype(bool)
            
            outliers_gains = np.logical_or(outliers_gains, outliers_k)

        n_outliers_gains = np.sum(np.logical_and(outliers_gains, np.logical_not(self.identifications['outliers'])))
        print(f"    + N(gain at limit): {n_outliers_gains} ({100*n_outliers_gains/self.identifications.shape[0]:.1f} %)")

        self.identifications['outliers_gains'] = outliers_gains
        self.identifications['outliers'] = np.logical_or(outliers_gains, self.identifications['outliers'])
        
        return self.identifications
    

    def _find_outliers_timedomain(self):
        """ Identify outliers whose step-response exceeds a certain z-value of the distribution of step-responses.
        """

        # outlier detection settings
        t_s = T_S
        z_score = self.timedomain_outlier_zscore
        T_test = self.timedomain_outlier_testperiod

        # simulated trajectories
        trajs, LD, pole_array, id_inliers = self._get_standardstepresponses()

        # find timedomain outliers
        traj_shape = trajs[0].shape
        N = int((T_test[1]-T_test[0])/t_s)
        t_begin = int(T_test[0]/t_s)
        t = np.arange(traj_shape[1]) * t_s
        t_end = traj_shape[1]

        if self.bikemodel == 'balancingrider':
            feature_map = {'p_x':0, 'p_y':1, 'psi':2, 'phi':5, 'delta':4}
            features_to_filter = ['psi'] #['delta']
            traj_array = np.zeros((len(features_to_filter)+1, traj_shape[1], len(trajs)))
        else:
            feature_map = {'p_x':0, 'p_y':1, 'psi':2}
            features_to_filter = ['psi']
            traj_array = np.zeros((len(features_to_filter)+1, traj_shape[1], len(trajs)))

        idx_features = [feature_map[f] for f in features_to_filter]

        for i in range(len(trajs)):
            traj_array[1:,:,i] = trajs[i][idx_features, :]
        traj_array[0, :, :] = LD

        mean_per_sample = np.mean(traj_array, axis=2)
        std_per_sample = np.std(traj_array, axis=2)

        interval_max_per_sample = mean_per_sample + z_score * std_per_sample
        interval_min_per_sample = mean_per_sample - z_score * std_per_sample

        outlier_i = []
        for i, f in enumerate(trajs):
            test_upper = traj_array[:,:,i] <= interval_max_per_sample
            test_upper = np.all(test_upper[:,t_begin:t_end])
            test_lower = traj_array[:,:,i] >= interval_min_per_sample
            test_lower = np.all(test_lower[:,t_begin:t_end])
            if not(test_upper and test_lower):
                outlier_i.append(i)
        outliers = np.zeros(len(trajs), dtype=bool)
        outliers[outlier_i] = True
        #poles['time-domain-outliers'] = outliers
         
        #test_loc_vals = np.array([evaluator.sample_metrics[m] for m in evaluator.test_loc_metrics]).T
        #outliers = np.zeros_like(test_loc_vals.shape[0], dtype=bool)

        #for i in range(test_loc_vals.shape[1]):
        #    LDvals = test_loc_vals[:,i]
        #    upper_lim = np.nanmean(LDvals) + z_score * np.nanstd(LDvals)
        #    lower_lim = np.nanmean(LDvals) - z_score * np.nanstd(LDvals)

        #    outliers = np.logical_or(outliers, np.logical_or(LDvals < lower_lim, LDvals > upper_lim))

        #outliers = np.logical_or(outliers, np.any(np.real(pole_array>-0.03),axis=1))
        outlier_i = np.argwhere(outliers).flatten()
        inlier_i = np.argwhere(np.logical_not(outliers)).flatten()

        # add outliers to table and print
        self.identifications['outliers_timedomain'] = self.identifications['sample_id'].isin(id_inliers.loc[outliers]['sample_id'])
        
        n_outliers_timedomain = np.sum(self.identifications['outliers_timedomain'])
        n_outliers_timedomain_new = np.sum(np.logical_and(self.identifications['outliers_timedomain'], np.logical_not(self.identifications['outliers'])))
        print(f"    + N(x(t) > {z_score}-sigma): {n_outliers_timedomain} ({100*n_outliers_timedomain/self.identifications.shape[0]:.1f} %)")

        self.identifications['outliers'] = np.logical_or(self.identifications['outliers_timedomain'], self.identifications['outliers'])

        #plot over time
        features_to_filter = ["LD"] + features_to_filter
        fig_t, axes_t = plt.subplots(len(features_to_filter), 1, sharex=True, layout='constrained')
        if not isinstance(axes_t, np.ndarray):
            axes_t = np.array([axes_t])
        for feat, ax in zip(features_to_filter, axes_t):
            for z in range(1,z_score+1):
                idx = features_to_filter.index(feat)
                ax.fill_between(t, 
                                mean_per_sample[idx,:] + z * std_per_sample[idx,:], 
                                mean_per_sample[idx,:] - z * std_per_sample[idx,:],
                                color=cmap.colors[0],
                                alpha=0.25)
            #line_inlier = ax.plot(t, mean_per_sample[idx,:], color = cmap.colors[0], label='inliers')
            line_inlier = ax.plot(t, traj_array[idx,:,inlier_i].T, color = cmap.colors[2], linewidth = 1, label='outliers')
            line_outlier = ax.plot(t, traj_array[idx,:,outlier_i].T, color = cmap.colors[6], linewidth = 1, label='outliers')

            ax.set_ylabel(feat)
            
        n_outliers = len(outlier_i)
        axes_t[-1].legend(handles=[l[0] for l in [line_inlier, line_outlier] if len(l) > 0])
        axes_t[-1].set_xlabel("t [s]")
        axes_t[0].set_title(f"Time-distribution of standard step responses with {n_outliers_timedomain} LD outliers exceeding {z_score}-sigma.")

        #plot in space
        fig_xy, axes_xy = plt.subplots(1, 1, layout='constrained')
        axes_xy.set_aspect('equal')
        for i in range(traj_array.shape[2]):
            if i in outlier_i: 
                line_outlier = axes_xy.plot(trajs[i][0,:].T, trajs[i][1,:].T, color = cmap.colors[6], linewidth = 1, label='outliers', zorder=1000)
            else:
                line_inlier = axes_xy.plot(trajs[i][0,:].T, trajs[i][1,:].T, color = cmap.colors[0], linewidth = 1, alpha=0.3, label='inliers')
        axes_xy.set_xlabel('x [m]')
        axes_xy.set_xlabel('y [m]')
        axes_xy.legend(handles=[l[0] for l in [line_inlier, line_outlier] if len(l) > 0])
        axes_xy.set_title(f"XY-Distribution of standard step responses with {n_outliers_timedomain} LD outliers exceeding {z_score}-sigma.")

        if self.save:
            
            if self.riderbike_model is not None:
                filepath = self.riderbike_model + "_"
            else:
                filepath = ""

            filepath_xy = os.path.join(self.paths["dir_out"], filepath+f"{self.OUTPUT_FNAMES["standard-stepresponses"]}_outliers_xy.png")
            fig_xy.set_size_inches(16.5, 4.5)
            fig_xy.savefig(filepath_xy)

            filepath_t = os.path.join(self.paths["dir_out"], filepath+f"{self.OUTPUT_FNAMES["standard-stepresponses"]}_outliers_t.png")
            fig_t.set_size_inches(16.5, 9.5)
            fig_t.savefig(filepath_t)


    def _argssort(self, poles, method):
        """ Sort poles by method."""

        if method == "magnitude":
            sort_idx = np.argsort(np.abs(poles))
        elif method == "argument":
            sort_idx = np.argsort(np.abs(np.angle(poles)))
        elif method == "frequency":
            sort_idx = np.argsort(np.abs(np.imag(poles)))
        elif method == "decay":
            sort_idx = np.argsort(np.abs(np.real(poles)))
        elif method == "none":
            sort_idx = np.arange(poles.size)
        else:
            raise RuntimeError(f"Invalid method: {self.method}")

        return sort_idx
    

    def _make_poles(self, real_poles, comp_poles):
        """ Makes poles from pole features and adds them to the id table. 
        """

        #id_keys = ['sample_id', 'participant', 'objective', 'v_mean', 'f_nom', 'outliers']
        #self.identifications = self.identifications[id_keys]

        #add real poles
        real_poles = np.real(real_poles).flatten()
        self.identifications['p0_real'] = real_poles

        #add imaginary poles
        comp_poles_real = np.real(comp_poles)
        comp_poles_imag = np.abs(np.imag(comp_poles))
        comp_poles_ang = np.abs(np.angle(comp_poles))
        comp_poles_mag = np.abs(comp_poles)

        for i in range(comp_poles.shape[1]):
            self.identifications[f'p{i+1}_real'] = comp_poles_real[:,i]
            self.identifications[f'p{i+1}_imag'] = comp_poles_imag[:,i]
            self.identifications[f'p{i+1}_mag'] = comp_poles_mag[:,i]
            self.identifications[f'p{i+1}_ang'] = comp_poles_ang[:,i]

        #self.identifications = self.identifications.drop(np.argwhere(np.isnan(self.identifications['p0_real'])).flatten())
    
    
    def _extract_poles(self):
        """
        Extract poles from the identification result
        """

        #get pole keys
        pole_keys = []
        pattern = r'pole\d'
        for k in self.identifications:
            if re.findall(pattern, k):
                pole_keys.append(k)
                self.identifications[k] = self.identifications[k].apply(lambda p: complex(p))
    
        #extract poles
        poles = self.identifications[pole_keys].to_numpy()

        return poles
    

    def _make_poleplot(self, pole_array, mark=None):
        """ Make a plot of poles. """

        if mark is None:
            mark = np.zeros(pole_array.shape[0], dtype=bool)

        #plot inliers
        fig = plt.figure(layout='constrained')
        gs = GridSpec(2,2, height_ratios=[4, 1], width_ratios=[4,1], figure=fig)
    
        ax = fig.add_subplot(gs[0,0])
        ax.set_title(f"Inlier poles (N={pole_array.shape[0]}) sorted by {self.method_complex}: {self.riderbike_model}")
        #ax.set_aspect("equal")
        ax.grid(True)
        ax.set_xlim(max(np.min(pole_array), -25), 0)

        ax_histreal = fig.add_subplot(gs[1, 0], sharex=ax)
        ax_histreal.set_xlabel("Re")
        ax_histreal.set_ylabel("counts")
        ax_histimag = fig.add_subplot(gs[0, 1], sharey=ax)
        ax_histimag.set_ylabel("Im")
        ax_histreal.set_xlabel("counts")

        # Scatter and hisogram plot
        sctr_kwargs = {"s": 35, "marker": ".", "alpha":0.5, "edgecolor": "none"}
        sctr_kwargs_mark = {"s": 35, "marker": "o", "alpha":0.5, "facecolor": "none", "edgecolor": tudcolors.get("rood")}
        histreal_kwargs = {"bins": 50, "alpha": 0.5}
        histimag_kwargs = {"bins": 100, "alpha": 0.5, "orientation":"horizontal"}

        for i in range(pole_array.shape[0]):   
            p = pole_array[i,:]           
            ax.scatter(np.real(p), np.imag(p), color=cmap.colors[i%len(cmap.colors)], **sctr_kwargs)
            ax.scatter(np.real(p), -np.imag(p), color=cmap.colors[i%len(cmap.colors)], **sctr_kwargs)

            if mark[i]:
                ax.scatter(np.real(p), np.imag(p), color=cmap.colors[i%len(cmap.colors)], **sctr_kwargs_mark)
                ax.scatter(np.real(p), -np.imag(p), color=cmap.colors[i%len(cmap.colors)], **sctr_kwargs_mark)

            
            ax_histreal.hist(np.real(p), color=cmap.colors[i%len(cmap.colors)], **histreal_kwargs)
            if np.any(np.imag(p) != 0.0):
                ax_histimag.hist(np.concatenate((np.imag(p), -np.imag(p))), color=cmap.colors[i%len(cmap.colors)], **histimag_kwargs)
          
            if not np.all(np.imag(p) == 0.0):
                i+=1

        return fig, ax, ax_histimag, ax_histreal

    def _make_outlier_poleplot(self, mark=None):
        """ Make a plot of outlier poles. """

        if mark is None:
            mark = np.zeros(self.identifications.shape[0], dtype=bool)

        #plot inliers
        fig = plt.figure(layout='constrained')
        gs = GridSpec(2,2, height_ratios=[4, 1], width_ratios=[4,1], figure=fig)
    
        ax = fig.add_subplot(gs[0,0])
        ax.set_title(f"Pole outliers per type: {self.riderbike_model}")
        #ax.set_aspect("equal")
        ax.grid(True)

        ax_histreal = fig.add_subplot(gs[1, 0], sharex=ax)
        ax_histreal.set_xlabel("Re")
        ax_histreal.set_ylabel("counts")
        ax_histimag = fig.add_subplot(gs[0, 1], sharey=ax)
        ax_histimag.set_ylabel("Im")
        ax_histreal.set_xlabel("counts")

        # Scatter and hisogram plot
        sctr_kwargs = {"s": 45, "marker": ".", "edgecolor": "none"}
        sctr_kwargs_mark = {"s": 45, "marker": "o", "facecolor": "none", "edgecolor": tudcolors.get("rood")}
        histreal_kwargs = {"bins": 50, "alpha": 0.5}
        histimag_kwargs = {"bins": 100, "alpha": 0.5, "orientation":"horizontal"}

        outlier_types = ['outliers_stability', 'outliers_timedomain', 'outliers_configuration', 'outliers_gains', 'outliers_objective'] #'outliers_response_time', 
        pole_keys = [f"pole{i}" for i in range(10) if f"pole{i}" in self.identifications]

        for i, outlier_type in enumerate(outlier_types):

            mask = self.identifications[outlier_type]

            p = np.array(self.identifications[mask][pole_keys])
            N = p.shape[0]
            p = p.flatten()

            if not np.any(np.isfinite(p)):
                continue

            ax.scatter(np.real(p), np.imag(p), color=cmap.colors[i], **sctr_kwargs, label=outlier_type + f"(N={N})")
            ax.scatter(np.real(p), -np.imag(p), color=cmap.colors[i], **sctr_kwargs)

            if np.any(mark[mask]):
                q = np.array(self.identifications[mask][pole_keys])[mark[mask]].flatten()
                ax.scatter(np.real(q), np.imag(q), **sctr_kwargs_mark)
                ax.scatter(np.real(q), -np.imag(q), **sctr_kwargs_mark)

            ax_histreal.hist(np.real(p), color=cmap.colors[i], **histreal_kwargs)
            ax_histimag.hist(np.concatenate((np.imag(p), -np.imag(p))), color=cmap.colors[i], **histimag_kwargs)
          
            if not np.all(np.imag(p) == 0.0):
                i+=1
        
        ax.legend()

        # maximum x limits
        x_max = min(ax.get_xlim()[1], 5)
        x_min = max(ax.get_xlim()[0], -15)
        ax.set_xlim(x_min, x_max) 
        ax_histreal.set_xlim(x_min, x_max)


        return fig, ax, ax_histimag, ax_histreal
    
    def plot_poles(self):
        """ Plot all poles. Creates a separate plots for inliers and outliers."""

        id_inliers = self.identifications[np.logical_not(self.identifications["outliers"])]
        id_outliers = self.identifications[self.identifications["outliers"]]

        poles_inliners = polefeaturetable_to_polearray(id_inliers)
        #poles_outliers = polefeaturetable_to_polearray(id_outliers)

        fig_in, ax_in, _, _ = self._make_poleplot(poles_inliners)
        fig_out, ax_out, _, _ = self._make_outlier_poleplot()

        if self.save:
            filepath_in = f"{self.OUTPUT_FNAMES['pole-sorting-result']}_{self.method_complex}_inliers.png"
            filepath_out = f"{self.OUTPUT_FNAMES['pole-sorting-result']}_{self.method_complex}_outliers.png"
            if self.riderbike_model is not None:
                filepath_in = self.riderbike_model + "_" + filepath_in
                filepath_out = self.riderbike_model + "_" + filepath_out
            filepath_in = os.path.join(self.paths['dir_out'], filepath_in)
            filepath_out = os.path.join(self.paths['dir_out'], filepath_out)

            fig_in.set_size_inches(16.5, 9.5)
            fig_in.savefig(filepath_in)

            fig_out.set_size_inches(16.5, 9.5)
            fig_out.savefig(filepath_out)


    def get_pole_feature_table(self):
        """ Returns a pole feature table including AngMag and ImRe pole features.

        Removes:
        - unstable poles
        - poles in the wrong format (i.e. number of real/complex poles)

        Marks as outliers:
        - temporal outliers

        Returns
        -------
        pole_table : pd.DataFrame
            The pole table.
        """
        #extract poles
        poles = self._extract_poles()

        #sort poles
        real_poles = []
        comp_poles = []
        outliers_configuration = np.zeros(poles.shape[0], dtype=bool)
        outliers_stability = np.zeros(poles.shape[0], dtype=bool)

        # identify real and complex poles
        for i_row in range(poles.shape[0]):
            real_poles_i = []
            comp_poles_i = []
            for p in poles[i_row,:]:
                if np.imag(p) == 0.0:
                    real_poles_i.append(p)
                else:
                    if np.conjugate(p) not in comp_poles_i:
                        comp_poles_i.append(p)

            real_poles_i = np.array(real_poles_i)
            comp_poles_i = np.array(comp_poles_i)

            #sort poles
            comp_poles_i = np.array(comp_poles_i)

            if comp_poles_i.size > 0:
                sort_idx_comp = self._argssort(comp_poles_i, method=self.method_complex)
                comp_poles_i = comp_poles_i[sort_idx_comp]

            # detect unstable poles
            if np.real(real_poles_i[0]) >= 0 or np.any(np.real(comp_poles_i) > 0):
                outliers_stability[i_row] = True

            # detect wrong pole configurations
            if len(real_poles_i) > 1:
                comp_poles_i = np.nan * np.ones_like(comp_poles[-1])
                real_poles_i = np.array([np.nan])
                outliers_configuration[i_row] = True

            real_poles.append(real_poles_i)
            comp_poles.append(comp_poles_i)

        # make poles and add to table
        real_poles = np.array(real_poles)
        comp_poles = np.array(comp_poles)
        self._make_poles(real_poles, comp_poles)
        
        # outliers
        self.identifications['outliers'] = np.zeros(self.identifications.shape[0], dtype=bool)
        
        print("Outlier detection:")

        #outliers stability
        n_outliers_stability = np.sum(outliers_stability)
        print(f"    N(unstable): {n_outliers_stability} ({100*n_outliers_stability/self.identifications.shape[0]:.1f} %)")

        self.identifications['outliers_stability'] = outliers_stability
        self.identifications['outliers'] = outliers_stability
                      
        #outliers configuration
        n_outliers_config = np.sum(np.logical_and(outliers_configuration, np.logical_not(self.identifications['outliers'])))
        print(f"    + N(invalid pole config): {n_outliers_config} ({100*n_outliers_config/self.identifications.shape[0]:.1f} %)")
        
        self.identifications['outliers_configuration'] = outliers_configuration
        self.identifications['outliers'] = np.logical_or(outliers_configuration, self.identifications['outliers'])

        #outliers identification pipeline
        #self._find_outliers_responsetime()
        self._find_outliers_objective()
        self._find_outliers_gainlimit()

        #outliers timedomain
        self._find_outliers_timedomain()

        n_outliers = np.sum(self.identifications['outliers'])
        print(f"    ---------------------")
        print(f"    TOTAL: {n_outliers} ({100*n_outliers/self.identifications.shape[0]:.1f} %)")

        if self.save:
            filepath = f"{self.riderbike_model}_{self.OUTPUT_FNAMES['pole-sorting-result']}_{self.method_complex}.csv"
            self.identifications.to_csv(os.path.join(self.paths['dir_out'], filepath), sep=";")

        return self.identifications 
    
class LogTransformer():
    """ A data transformer applying a log-shift transformation to the given data:
    
    y = log(x-1)

    """

    def __init__(self, alpha=0.9):
        """ Create a logshift transformer object.

        Parameters
        ----------
        alpha : float, optional
            Factor for fitting the shift parameter. a will be chosen as alpha * min(X).
            Must be in [0,1]. Default is 0.9
        """
        self.alpha = alpha
        if alpha <= 0.0 or alpha >= 1.0:
            raise ValueError(f"alpha must be in ]0,1[. Instead it was {alpha}!")

        self.a_ = None  
        self.sign_ = None


    def _check_X(self, X, limit=None):
        if limit is None:
            limit = self.a_
        if np.any((X - limit) <= 0):
            raise ValueError(f"All elements of X must be larger then {limit}!")


    def fit(self, X, y=None):
        """ Fit the logshift transformer.

        Determines:
            - required sign to enable log
            - a from the smallest value in (postitive) X

        Return
        ------
        LogTransformer
            The fitted logshift transformer. 
        """

        X = np.asarray(X)

        # find sign 
        self.sign_ = np.sign(X[0,:]).reshape((1,-1))
        X = X * self.sign_

        self._check_X(X, limit=0)
        
        # find shift parameter
        self.a_ = self.alpha * np.min(X, axis=0)
        self.a_ = self.a_.reshape(1, X.shape[1])

        return self


    def transform(self, X):
        """ Apply the logshift trafo to the data in X.
        """

        X = np.asarray(X)
        
        X = X * self.sign_
        self._check_X(X)
        if self.a_ is None:
            raise RuntimeError("The transformer has not been fitted yet.")
        return np.log(X - self.a_)
    
    def inverse_transform(self, X):
        """ Apply the inverse logshift trafo to the data in X.
        """

        X = np.asarray(X)

        if self.a_ is None:
            raise RuntimeError("The transformer has not been fitted yet.")
        Y = np.exp(X) + self.a_
        Y = Y * self.sign_

        return Y


class PreprocessingPipeline():
    """ A pipeline applying multiple preprocessing steps after another."""

    POWER_TRANSFORMS = ("yeo-johnson", "box-cox", "none")

    def __init__(self, feature_set, features, normalize=True, log_transform=True, power_transform="yeo-johnson", save=False, dir_out=None, tag=None):
        """ Create a Preprocessing pipeline object. 
        
        Parameters
        ----------
        feature_set : str
            Name of any of the feature sets (See PoleModel).
        features : list
            The features of the feature set (see PoleModel).
        normalize : bool, optional
            Normalize the data. Default is True
        log_transform : bool, optional
            Apply log transform. Default is True
        power_transform : str, optional
            Apply a power transformation. Can be "yeo-johnson", "box-cox", "none". 
            Default is "yeo-johnson"
        save : bool, optional
            If True, automatically saves a plot form fitting. 
        dir_out : str, optional
            The directory for saving the plot.
        tag : str, optional
            A tag for the name of the figure.
        """
        
        #Transforms
        self.normalize = normalize

        if power_transform in self.POWER_TRANSFORMS:
            self.power_transform = power_transform
        else:
            raise NotImplementedError(f"Power transformation '{power_transform}' not implemented! Choose any of {self.POWER_TRANSFORMS}.")
    
        self.log_transform = log_transform

        #
        self.transformers_ = []
        self.is_fitted_ = False

        #Output
        self.feature_set = feature_set
        self.features = features
        self.save = save
        self.dir_out = dir_out
        self.tag = tag

        method_str = ""
        if power_transform != 'none':
            method_str+=f"{power_transform}-pt, "
        if normalize:
            method_str+=f"normalized, "
        method_str = method_str[:-2]  

        self.method_str = method_str

    def from_parameters(feature_set, features, normalize=False, power_transform="yeo-johnson", log_transform=False, 
                        power_transform_params={}, standard_scaler_params={}, log_transform_params={}, 
                        save=False, dir_out=None, tag=None):
        """ Create a ProcessingPipeline object from known parameters.
        """
        if power_transform !="yeo-johnson":
            raise NotImplementedError("Initializing a PreprocessingPipeline with other power transformation then yeo-johnson is not implemented!")

        pipe = PreprocessingPipeline(feature_set, features, normalize=normalize, power_transform=power_transform, 
                                     log_transform=log_transform, save=save, dir_out=dir_out, tag=tag)
        
        pipe.method_list = [f'{pipe.power_transform}-power-transform']
        pipe.n_features = len(pipe.features)

        if log_transform:
            pipe.log_transform_features_ = np.array(log_transform_params["log_transform_features"])
            pipe.transformers_.append(LogTransformer())
            pipe.transformers_[-1].a_ = np.array(log_transform_params["a"])
            pipe.transformers_[-1].sign_ = np.array(log_transform_params["sign"])

        if normalize:
            scaler = StandardScaler()
            scaler.mean_ = np.array(standard_scaler_params["mean"])
            scaler.scale_ = np.array(standard_scaler_params["scale"])
            scaler.var_ = np.array(standard_scaler_params["scale"])**2
            scaler.n_features_in_ = len(standard_scaler_params["mean"])
            scaler.n_samples_seen_ = standard_scaler_params["n_samples_seen"]

        if pipe.power_transform != "none":
            pipe.transformers_.append(PowerTransformer(method=power_transform, standardize=normalize))
            #pipe.transformers_.append(QuantileTransformer(output_distribution='normal'))
            pipe.transformers_[-1].lambdas_ = np.array(power_transform_params["lambdas"])
            pipe.transformers_[-1].n_features_in_ = len(power_transform_params["lambdas"])
            if normalize:
                pipe.transformers_[-1]._scaler = scaler
        else:
            if normalize:
                pipe.transformers_.append(scaler)
            #if normalize:
            #    pipe.transformers_.append(scaler)




        pipe.is_fitted_ = True

        return pipe
        
    def _get_log_transformation(self, X):
        """ Get a y = log(x-a) transformer. 
        """

        # find features suitable for log transofrmation. 
        pattern = r"p\d_(.{1,5})"
        self.log_transform_features_ = []
        for i, feat in enumerate(self.features):
            match = re.findall(pattern, feat)
            if len(match)>0:
                if match[0] in ['real', 'mag']:
                    self.log_transform_features_.append(i)
        self.log_transform_features_ = np.array(self.log_transform_features_)

        if len(self.log_transform_features_) == 0:
            raise RuntimeError(f"The log transformer didn't find any 'real' or 'mag' features in {self.features}!")

        lt = LogTransformer(alpha=0.9).fit(X[:,self.log_transform_features_])
        X_transformed_lt = lt.transform(X[:,self.log_transform_features_])
        X_transformed = X.copy()
        X_transformed[:,self.log_transform_features_] = X_transformed_lt

        return X_transformed, lt

    def _get_power_transformation(self, X):
        """ Get a power transformer fitted to the data X and plot the transformation results. 
        """
        pt = PowerTransformer(standardize=self.normalize).fit(X)
        X_transformed = pt.transform(X)

        return X_transformed, pt
    

    def _get_standard_scaler(self, X):
        """ Get a standard scaler fitted to the data X and plot the transformation results. 
        """
        scaler = StandardScaler().fit(X)
        X_transformed = scaler.transform(X)
        
        return X_transformed, scaler
    

    def _plot_transformation(self, X_list, method_list):
        """ Plot a histogram comparison between transformed and non-transformed data
        """

        fig, axes = plt.subplots(len(X_list), X_list[0].shape[1], sharey=True, layout='constrained')
        axes = axes.reshape(len(X_list), X_list[0].shape[1])

        hist_kwargs = {"bins": 50}

        for i, X in enumerate(X_list):
            for j in range(X.shape[1]):
                axes[i,j].hist(X[:,j], color = cmap(i), **hist_kwargs)
            #axes[i,j].hist(X[:,i], color = cmap(i), **hist_kwargs)
                
            axes[i,0].set_ylabel("counts")
            axes[i, int(np.floor(X.shape[1]/2))].set_title(method_list[i])

        for j in range(X.shape[1]):
            axes[-1,j].set_xlabel(f'{self.features[j]}')
            
        fig.suptitle(f"Preprocessing pipeline on {self.feature_set} input data")

        if self.save:
            method_str = ""
            for m in method_list[1:]:
                method_str += f"{m}-"
            method_str = method_str[:-1]
            
            filepath = f"feature-trafo_{self.feature_set}_{method_str}.png"
            if self.tag is not None:
                filepath = self.tag + "_" + filepath
            filepath = os.path.join(self.dir_out, filepath)

            fig.set_size_inches(16.5, 9.5)
            fig.savefig(filepath)


    def fit_transform(self, X):
        """ Fit the pipeline to the data X and apply transformation to X. """

        self.n_features = X.shape[1]

        #X[:,1] = np.log(-X[:,1])
        #X[:,3] = np.log(-X[:,3])
        X_list = [X]
        method_list = ["orignal"]

        if self.log_transform:
            X_transformed, lt = self._get_log_transformation(X_list[-1])
            X_list.append(X_transformed)
            method_list.append(f'log-shift-transform')
            self.transformers_.append(lt)

        if self.power_transform != 'none':
            X_transformed, pt = self._get_power_transformation(X_list[-1])
            X_list.append(X_transformed)
            method_list.append(f'{self.power_transform}-power-transform')
            self.transformers_.append(pt)

        elif self.normalize:
            X_transformed, scaler = self._get_standard_scaler(X_list[-1])
            X_list.append(X_transformed)
            method_list.append(f'standard-scaling')
            self.transformers_.append(scaler)

        self._plot_transformation(X_list, method_list)

        return X_list[-1]


    def transform(self, X, sparse_column_indices=None):
        """ Apply the transformation of the preprocessing pipeline to the data X
        
        Parameters
        ----------
        sparce_column_indices : array-like
            Feature indices to apply the transformation to in case of sparse data.
        """

        # expand column-sparse sparse data
        if sparse_column_indices is not None:
            sparse_column_indices = np.array(sparse_column_indices).flatten()
            X_transform = np.zeros((X.shape[0], self.n_features))
            for i,j in enumerate(sparse_column_indices):
                X_transform[:,j] = X[:,i].flatten()
        else:
            X_transform = X.copy()


        #transform
        for trafo in self.transformers_:
            if isinstance(trafo, LogTransformer):
                #if logtrafo, only apply to suitable features
                for i in range(self.n_features):
                    if np.any(i==self.log_transform_features_) and not np.any(sparse_column_indices==i) and sparse_column_indices is not None:
                        idx_in_trafo = np.argwhere(i==self.log_transform_features_).flatten()
                        X_transform[:,i] = trafo.sign_[0,idx_in_trafo] * trafo.a_[0,idx_in_trafo] * 2
                X_transform[:,self.log_transform_features_] = trafo.transform(X_transform[:,self.log_transform_features_])
            else:
                X_transform = trafo.transform(X_transform)

        # reduce to column-sparse data
        if sparse_column_indices is not None:
            X_transform = X_transform[:,sparse_column_indices]

        return X_transform


    def inverse_transform(self, X, sparse_column_indices=None):
        """ Apply the inverse transformation of the preprocessing pipeline to the data X
        
        Parameters
        ----------
        sparce_column_indices : array-like
            Feature indices to apply the transformation to in case of sparse data.
        """
        
        # expand column-sparse sparse data
        if sparse_column_indices is not None:
            X_transform = np.zeros((X.shape[0], self.n_features))
            for i,j in enumerate(sparse_column_indices):
                X_transform[:,j] = X[:,i].flatten()
        else:
            X_transform = X.copy()

        #transform
        for trafo in reversed(self.transformers_):
            if isinstance(trafo, LogTransformer):
                #if logtrafo, only apply to suitable features
                X_transform[:, self.log_transform_features_] = trafo.inverse_transform(X_transform[:,self.log_transform_features_])
            else:
                X_transform = trafo.inverse_transform(X_transform)

        # reduce to column-sparse data
        if sparse_column_indices:
            X_transform = X_transform[:,sparse_column_indices]

        return X_transform
    


class PoleModel():
    """ A clase to fit predictive pole models. 
    """

    PREDEFINED_FEATURE_SETS = {
                     'ImRe5': [["p0_real", "p1_real", "p1_imag", "p2_real", "p2_imag"]],
                     'ImRe5GivenV': [["v_mean", "p0_real", "p1_real", "p1_imag", "p2_real", "p2_imag"], "v_mean"],
                     'AngMag5': [["p0_real", "p1_mag", "p1_ang", "p2_mag", "p2_ang"]],
                     'AngMag5GivenV': [["v_mean", "p0_real", "p1_mag", "p1_ang", "p2_mag", "p2_ang"], "v_mean"],
                     'Re1': [["p0_real"]],
                     'Re1GivenV': [["v_mean", "p0_real",], "v_mean"],
                     }
    
    REQUIRED_PATHS = ["filepath_partition", "filepath_sorted_poles", "dir_out"]

    SUBDIRS = {"output-dirname": "pole-modeling"}

    OUTPUT_FNAMES = {"gridsearch-results": "gridsearch",
                     "marginal-distributions": "distribution-model",
                     "model-export": "pole-model-params"}


    def __init__(self,
                 paths, 
                 gridsearch_selection_metric='NLL', 
                 normalization=False, 
                 power_transformation='yeo-johnson',
                 feature_set='ImRe5',
                 n_gmm_inits=100,
                 riderbike_model=None,
                 pole_table=None,
                 save=True,
                 from_data=True,
                 random_state=None):
        
        #data
        self.paths = self._check_paths(paths, save, from_data, pole_table is not None)
        self.riderbike_model = riderbike_model
        if from_data:
            self._load_data(pole_table)

        #features
        self.feature_set, self.features, self.feature_cond = self._check_feature_set(feature_set)

        #preprocessing
        self.normalize = normalization
        self.pt_type = power_transformation 
        self.pp_pipeline = PreprocessingPipeline(self.feature_set, 
                                    self.features, 
                                    normalize=self.normalize, 
                                    power_transform=self.pt_type, 
                                    save=save, 
                                    dir_out=self.paths['dir_out'],
                                    tag=self.riderbike_model)

        #model
        self._n_gmm_inits = n_gmm_inits
        self.is_fitted_ = False
        self.random_state = random_state

        #gridsearch
        self.gs_sel_metric = gridsearch_selection_metric
        
        #output
        self.save=save

    
    def _check_paths(self, paths, save, from_data, has_pole_table):
        """ Check that all required paths are supplied and exist. """

        required_paths = []
        if not has_pole_table and from_data:
            required_paths.append(self.REQUIRED_PATHS[1])
        if from_data:
            required_paths.append(self.REQUIRED_PATHS[0])
        if save:
            required_paths.append(self.REQUIRED_PATHS[2])

        for p in required_paths:
            if p not in paths:
                raise ValueError(f"Path to {p} missing in 'paths'. 'paths' must have at least {required_paths}.")
            if 'filepath' in p:
                if not os.path.isfile(paths[p]):
                    raise IOError(f"Can't find file '{p}' at {paths[p]}.")
            elif 'dir' in p:
                if paths[p] is None:
                    raise ValueError(f"Path {p} mustn't be None! Provide a valid path.")
                if not os.path.isdir(paths[p]):
                    raise IOError(f"Can't find directory '{p}' at {paths[p]}.")
        
        # if necessary, make output directory
        if not os.path.basename(os.path.normpath(paths['dir_out'])) == self.SUBDIRS["output-dirname"]:
            paths['dir_out'] = os.path.join(paths['dir_out'], self.SUBDIRS["output-dirname"])
        if not os.path.isdir(paths['dir_out']):
            os.makedirs(paths['dir_out'])

        return paths

    def _check_feature_set(self, feature_set):
        """Check that the selected feature set is valid."""
        
        valid_keys = list(self.PREDEFINED_FEATURE_SETS.keys())

        if isinstance(feature_set, str):
            if feature_set not in self.PREDEFINED_FEATURE_SETS:
                raise ValueError(f"If a String, 'feature_set' must be any of the predefined features {valid_keys}, not '{feature_set}'")
            
            feature_set_name = feature_set
            features = self.PREDEFINED_FEATURE_SETS[feature_set][0]
            if len(self.PREDEFINED_FEATURE_SETS[feature_set])>1:
                feature_cond = self.PREDEFINED_FEATURE_SETS[feature_set][1]
            else:
                feature_cond = ""
        else:
            raise NotImplementedError((f'Feature sets other then the predifined sets are '
                                       f'currently not implemented! Choose any of {valid_keys}'))
        
        self._make_tex_polelabels(features)
        
        return feature_set_name, features, feature_cond


    def _make_tex_polelabels(self, features):
        """Make pole labels in TeX maths format
        """
        self.tex_feature_labels = []
        for f in features:
            n = f[1]

            if n == '0':
                nstr = "2"
            if n == '1':
                nstr = "1, 3"
            if n == '2':
                nstr = "0, 4"

            if 'real' in f:
                l = r"$\Re(s_{"+nstr+r"})$"
            elif 'imag' in f:
                l = r"$|\Im(s_{"+nstr+r"})|$"
            elif 'ang' in f:
                l = r"$\varphi_"+str(n)+r"$"
            elif 'mag' in f:
                l = r"$r_"+str(n)+r"$"
            elif 'v_mean' in f:
                l = r"$\bar{v}$ [$m~s^{-1}$]"
            else:
                l = f
            self.tex_feature_labels.append(l)


    def _init_gmm(self, n_components, covariance_type):
        """Init a (conditional) GMM for the given hyperparamters.
        """ 

        gmm_kwargs = dict(n_init=self._n_gmm_inits,
                          n_components=n_components,
                          covariance_type=covariance_type)
        
        if self.feature_cond != '':
            gmm_kwargs["feature_index_given"] = self.features.index(self.feature_cond)
            return ConditionalGaussianMixture(**gmm_kwargs), score_conditional_gmm
        else:
            return GaussianMixture(**gmm_kwargs), score_gmm


    def _init_gmm_from_params(self, means, covariances, weights, random_state):
        """Init a (conditional) GMM from known parameters
        """

        if self.feature_cond != '':
            gmm = ConditionalGaussianMixture.from_parameters(means, covariances, weights, 
                                                             self.features.index(self.feature_cond),
                                                             random_state=random_state)
            score_func = score_conditional_gmm
            gmm.feature_indices_marginals = [i for i in np.arange(len(self.features)) if i != self.features.index(self.feature_cond)]
        else:
            gmm = GaussianMixture.from_parameters(means, covariances, weights, random_state=random_state)
            score_func = score_gmm

        return gmm, score_func


    def _check_conditional_features(self, conditional_features):
        """ Check if the conditional features appear in the given features.

        UNUSED.
        """
        
        missing_feature_error = ValueError(("The conditional features must appear in the feature list" 
                                           "of the corresponding feature set or, for non-conditional" 
                                           "models, be empty strings!"))
        
        for key in conditional_features:
            if key not in self.features:
                continue
            if conditional_features[key] not in self.features[key]:
                raise missing_feature_error
        
        for key in self.features:
            if key not in conditional_features:
                conditional_features[key] = ''

        return conditional_features
    
    
    def _load_data(self, pole_table):
        """ Load the data.
        """

        # load fixed partitioning
        self.partition = read_yaml(self.paths["filepath_partition"])

        if pole_table is None:
            # load poles
            pole_table = pd.read_csv(self.paths["filepath_sorted_poles"])

        if 'outliers_all' in pole_table:
            outliers = pole_table['outliers_all'] 
        else:
            outliers = pole_table['outliers'] 
        
        self.pole_table_outliers = pole_table[outliers]
        self.pole_table_inliers = pole_table[np.logical_not(outliers)]


    def _calibrate_variance_scale(self, X_train):

        var_scale = np.linspace(0.2, 1.0, 25)
        n_calib_samples = 10000
        
        alpha = 0.05
        n_quantile = int(round(X_train.shape[0] * (alpha)))
        if n_quantile == 0:
            raise RuntimeError(f"Not enough samples for alpha={alpha} calibration!")
        
        calib_score = np.zeros_like(var_scale)

        gmm_0, score_func = self._init_gmm(n_components=self.hyperparameters_['n_components'], 
                                    covariance_type=self.hyperparameters_['cov_type'], 
                                    var_scale=1.0)
        gmm_0.fit(X_train)
        nll_train = gmm_0.score_samples(X_train)
        worst_samples_train = np.argsort(nll_train)[-n_quantile:]
        nll_limit = np.min(nll_train[worst_samples_train])

        for i, s in enumerate(var_scale): 
            gmm, score_func = self._init_gmm(n_components=self.hyperparameters_['n_components'], 
                                    covariance_type=self.hyperparameters_['cov_type'], 
                                    var_scale=s)
            gmm = gmm.fit(X_train)   
            X_calib, _ = gmm.sample(n_samples=n_calib_samples)

            nll_calib = gmm.score_samples(X_calib)
            calib_score[i] = np.sum(nll_calib>nll_limit)/n_calib_samples

        best_calib = np.argmin(np.abs(calib_score-alpha))
        s_best = var_scale[best_calib]
        calib_score_best = calib_score[best_calib]

        print(f"    Variance Scale calibration at alpha={alpha} ({n_quantile} worst training samples): s={s_best}, score={calib_score_best}")

        self.hyperparameters_['var_scale'] = s_best
        self.scores_val_['variance_scale_calibration'] = calib_score_best

    def get_datasets(self):

        # get dataset
        X = self.pole_table_inliers[self.features].to_numpy()

        idx_train = self.pole_table_inliers['sample_id'].isin(self.partition["train"])
        idx_test = self.pole_table_inliers['sample_id'].isin(self.partition["test"])
        
        X = self.pp_pipeline.fit_transform(X)

        X_train = X[idx_train,:]
        X_test = X[idx_test,:]

        self.n_features_ = X_train.shape[1]
        self.n_samples_test_ = X_test.shape[0]
        self.n_samples_train_ = X_train.shape[0]

        return X_train, X_test
        

    def fit_optimize(self, 
                     range_gmm_components=[1,5], 
                     k_crossval=10,
                     covariance_types=["full", "tied", "diag", "spherical"]):
        """ Fit the pole model. Finds optimal hyperparmeters within the given ranges using cross-validation."""

        print(f"Fitting {self.feature_set} pole model on {self.riderbike_model} poledata:")
        self.k_crossval_ = 10
        
        #score dict
        model_scores = {
            "normalization": [],
            "power-transform": [],
            "cov_type": [],
            "n_components": [],
            "BIC": [],
            "AIC": [],
            "NLL": []}

        # get data
        X_train, X_test = self.get_datasets()


        # Grid-Search based model optimization
        for cov_type in covariance_types:
            print(" "*100, end="\r")
            for n in range(range_gmm_components[0], range_gmm_components[1]):
                print(f"    Running Gridsearch: covariance_type = {cov_type}, n_components = {n}", end="\r")
                # fit and validate a Gaussian Mixture Model with two components using cross-validation
                gmm, score_func = self._init_gmm(n, cov_type)
                scores = cross_validate(gmm, X_train, scoring=score_func, cv=k_crossval, error_score='raise')
                
                model_scores["normalization"].append(self.normalize)
                model_scores["power-transform"].append(self.pt_type)
                model_scores["n_components"].append(n)
                model_scores["BIC"].append(np.mean(scores["test_BIC"]))
                model_scores["AIC"].append(np.mean(scores["test_AIC"]))
                model_scores["NLL"].append(np.mean(scores["test_NLL"]))
                model_scores["cov_type"].append(cov_type)

        # Identify best hyperparamters
        self.gridsearch_scores_ = pd.DataFrame.from_dict(model_scores)
        best = self.gridsearch_scores_[model_scores[self.gs_sel_metric]==np.min(model_scores[self.gs_sel_metric])]
        self.scores_val_ = best[["BIC", "AIC", "NLL"]].iloc[0].to_dict()
        self.hyperparameters_ = best[["n_components", "cov_type"]].iloc[0].to_dict()

        # fit and test a model on the full training dataset with the best hyperparameters

        self.gmm_, score_func = self._init_gmm(n_components=self.hyperparameters_['n_components'], 
                                     covariance_type=self.hyperparameters_['cov_type'])
        self.gmm_ = self.gmm_.fit(X_train)   
        self.is_fitted_ = True     

        self.scores_test_ = score_func(self.gmm_, X_test)

        # plot best model
        self.plot_marginals(X_train, X_test, k_crossval)
        self.plot_gridsearch()

        print(f"    Finished gridsearch with best results covariance_type={self.hyperparameters_["cov_type"]} and n_components={self.hyperparameters_["n_components"]} achieving NLL={self.scores_val_["NLL"]:.4f}")

        return self
    

    def sample(self, n_samples=1, X_given=None, shuffle=True):
        """ Sample from the fitted distribution. 

        If PoleModel is a conditional model. A value to be conditioned on must be given. 

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to be drawn
        X_given : list-like
            List of n_given values of the conditional feature to be conditioned on. 
        shuffle : bool, optional
            Shuffle the returned samples. Default is True

        Returns
        -------
        samples : array-like
            Array of samples shaped [n_samples, n_features]. If conditional and n_given>1, the shape is
            Array of samples shaped [n_given, n_samples, n_features].
        """

        if self.feature_cond != '':
            if X_given is None:
                raise ValueError("Specify values for {self.feature_cond} to be conditioned on to sample poles!")
            X_given = np.array(X_given)
            x_given_temp = np.zeros((X_given.size, self.n_features_))
            x_given_temp[:,self.features.index(self.feature_cond)] = X_given.flatten()
            x_given_temp = self.pp_pipeline.transform(x_given_temp, sparse_column_indices=[self.features.index(self.feature_cond)])
            X_given = x_given_temp[:,self.features.index(self.feature_cond)]

            samples, labels = self.gmm_.sample(n_samples=n_samples, X_given=X_given)
        else:
            samples, labels = self.gmm_.sample(n_samples)

        if not np.all(np.isfinite(samples)):
            raise RuntimeError("Sampling error!")
        
        indices = [i for i, f in enumerate(self.features) if f != self.feature_cond]
        samples_out = self.pp_pipeline.inverse_transform(samples, sparse_column_indices=indices)
        
        # the sampled values may violate the valid range of the yeo-johnson inverse transform. Resample invalid values. 
        i = 0
        while not np.all(np.isfinite(samples_out)):
            missing_samples = np.logical_not(np.all(np.isfinite(samples_out), axis=1))
            n_missing_samples = np.sum(missing_samples)

            if self.feature_cond != '':
                new_samples, new_labels = self.gmm_.sample(n_samples=n_missing_samples, X_given=X_given)
            else:
                new_samples, new_labels = self.gmm_.sample(n_missing_samples)

            new_samples = self.pp_pipeline.inverse_transform(new_samples, sparse_column_indices=indices)
            samples_out[missing_samples, :] = new_samples
            labels[missing_samples] = new_labels

            i+=1
            if i>100:
                raise RuntimeError("Sampling error!")
    
        if not np.all(np.isfinite(samples_out)):
            raise RuntimeError("Sampling error!")

        #shuffle
        if shuffle:
            rng = np.random.default_rng(seed=self.random_state)
            shuffle_idx = np.arange(n_samples)
            rng.shuffle(shuffle_idx)
            
            samples_out = samples_out[shuffle_idx,:]
            labels = labels[shuffle_idx]

        if not np.all(np.isfinite(samples_out)):
            raise RuntimeError("Sampling error!")

        return samples_out, labels
    

    def sample_poles(self, n_samples=1, X_given=None, ensure_stable=True):
        """ Sample poles from the fitted distribution. 

        If PoleModel is a conditional model. A value to be conditioned on must be given. 

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to be drawn
        X_given : list-like
            List of n_given values of the conditional feature to be conditioned on. 

        Returns
        -------
        samples : array-like
            Array of pole samples shaped [n_samples, n_poles]. 
        """

        if n_samples == 0:
            return np.array([[]]), np.array([])


        if X_given is not None:
            if not isinstance(X_given, float):
                raise TypeError(f"X_given must be 'float', not '{type(X_given).__name__}'!")
            
        features = [f for f in self.features if f != self.feature_cond]
        feature_indices = [i for i, f in enumerate(self.features) if f != self.feature_cond]

        if 'AngMag' in self.feature_set:
            feat = 'AngMag'
        else:
            feat = 'ImRe'
            
        def _sample(n):
            samples, labels = self.sample(n_samples=n, X_given=[X_given])
            pole_table = pd.DataFrame(samples, columns=features)
            pole_array = polefeaturetable_to_polearray(pole_table, feat)
            return pole_array, labels

        # draw initial sample
        pole_array, labels = _sample(n_samples)

        if ensure_stable:
            n_iter = 0
            n_iter_max = 1000
            while np.any(np.real(pole_array)>0):
                mask_unstable = np.any(np.real(pole_array)>0, axis=1)
                extra_poles = _sample(np.sum(mask_unstable))
                pole_array[mask_unstable,:] = extra_poles
                n_iter+=1

                if n_iter > n_iter_max:
                    raise TimeoutError(f"Couldn't find {n_samples} stable poles after {n_iter_max} draws!")

        return pole_array, labels

    def get_component_means(self, X_given=None):
        """ Return the component means. 

        If PoleModel is a conditional model. A value to be conditioned on can be given.

        Parameters
        ----------
        X_given : list-like
            List of n_given values of the conditional feature to be conditioned on. 

        Returns
        -------
        mean_components : array-like
            Array of mean_features shaped [n_samples, n_features(, n_given)]. If X_given is None, the array is 2D.  
        x_cond : array like
            Array of conditional values corresponding to the component means. Only returned if the pole model is conditional
            but no values to be conditioned on are given. 
        """

        if X_given is not None:
            if not isinstance(X_given, (float, list, tuple, np.ndarray)):
                raise TypeError(f"X_given must be 'float' or 'array-like', not '{type(X_given).__name__}'!")
            if isinstance(X_given, float):
                X_given = np.array([X_given])
            else:
                X_given = np.array([X_given]).flatten()
            
            x_given_temp = np.zeros((X_given.size, self.n_features_))
            x_given_temp[:,self.features.index(self.feature_cond)] = X_given.flatten()
            x_given_temp = self.pp_pipeline.transform(x_given_temp, sparse_column_indices=[self.features.index(self.feature_cond)])
            X_given = x_given_temp[:,self.features.index(self.feature_cond)]
            
        features = [f for f in self.features if f != self.feature_cond]
        feature_indices = [i for i, f in enumerate(self.features) if f != self.feature_cond]

        if 'AngMag' in self.feature_set:
            feat = 'AngMag'
        else:
            feat = 'ImRe'

        if X_given is not None:
            component_means = []
            for x_given in X_given:
                means_x = self.gmm_._get_conditional_gmm(x_given).means_.reshape((-1,len(features)))
                means_x = self.pp_pipeline.inverse_transform(means_x, sparse_column_indices=feature_indices)

                #pole_table = pd.DataFrame(means_x, columns=features)
                #pole_array = polefeaturetable_to_polearray(pole_table, feat)

                component_means.append(means_x)

            component_means = np.array(component_means).transpose((1, 2, 0))
        else:
            component_means = self.gmm_.means_
            component_means = self.pp_pipeline.inverse_transform(component_means)
            #pole_table = pd.DataFrame(means, columns=features)
            #component_means = polefeaturetable_to_polearray(pole_table, feat)

            if self.feature_cond != '':
                x_cond = component_means[:,self.features.index(self.feature_cond)]
                component_means = component_means[:,feature_indices]
                return component_means, x_cond

        return component_means

    def get_component_mean_poles(self, X_given=None):
        """ Return the component means as complex poles. 

        If PoleModel is a conditional model. A value to be conditioned on can be given.

        Parameters
        ----------
        X_given : list-like
            List of n_given values of the conditional feature to be conditioned on. 

        Returns
        -------
        mean_poles : array-like
            Array of mean poles shaped [n_samples, n_poles(, n_given)]. If X_given is None, the array is 2D.  
        x_cond : array like
            Array of conditional values corresponding to the component means. Only returned if the pole model is conditional
            but no values to be conditioned on are given. 
        """

        if self.feature_cond != '' and X_given is None:
            component_means, x_cond = self.get_component_means(X_given=X_given)
        else:
            component_means = self.get_component_means(X_given=X_given)
        mean_poles = np.zeros_like(component_means, dtype=complex)

        features = [f for f in self.features if f != self.feature_cond]
        feature_indices = [i for i, f in enumerate(self.features) if f != self.feature_cond]

        if 'AngMag' in self.feature_set:
            feat = 'AngMag'
        else:
            feat = 'ImRe'

        if component_means.ndim > 2:
            for i in range(component_means.shape[2]):
                pole_table = pd.DataFrame(component_means[:,:,i], columns=features)
                mean_poles[:,:,i] = polefeaturetable_to_polearray(pole_table, feat)
        else:
            pole_table = pd.DataFrame(component_means, columns=features)
            mean_poles = polefeaturetable_to_polearray(pole_table, feat)

        if self.feature_cond != '' and X_given is None:
            return mean_poles, x_cond
        else:
            return mean_poles
        

    def get_component_mean_function_params(self):
        """ Return the component means as a function of speed

        WARNING: This will be wrong if the means are not linear over speed! Check the marginal distribution plot 
        to verify!


        Returns
        ------
        array_like
            An array of the parameters of function linear in speed is returned. The array is shaped
            (n_components, n_features, 2), with the last dimension representing intercept and coefficient. 
        """

        if isinstance(self.gmm_, ConditionalGaussianMixture):
            speeds_cond = np.linspace(1.5,5.5,250)
            means = self.get_component_means(speeds_cond)

            regs = []
            scores = []
            print("Fitting a linear function of speed to the component means.")
            print(f"score per component: ")
            if means.ndim != 3:
                raise NotImplementedError("Not implemented for models with n_components=1!")
            for i in range(means.shape[0]):
                means_i = means[i, :, :].T
                reg = LinearRegression().fit(speeds_cond.reshape(-1,1), means_i)
                score = reg.score(speeds_cond.reshape(-1,1), means_i)
                scores.append(score)
                regs.append(reg)
                print(f" component {i}: R2 = {score:.2f}")

            if np.any(np.array(scores) < 0.9):
                print(f"   Fit resulted in an unsatifactory R2. Confirm that the speed"
                      f" dependency of the component means in linear by looking at the plot of the 2D marignals!")
                
            return np.stack([np.c_[reg.intercept_, reg.coef_.flatten()] for reg in regs], axis=0)
        else:
            means = self.get_component_means()
            return means


    def plot_gridsearch(self):
        """ Plot the gridsearch model selection results.
        """

        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before plotting by calling 'fit_optimize()")

        # get params
        covariance_types = np.unique(self.gridsearch_scores_['cov_type'])
        metrics = np.unique(list(self.scores_val_.keys()))

        # make axes
        fig, axes = plt.subplots(1, metrics.size, layout="constrained")

        # plot results
        for metric, ax in zip(metrics, axes):
            for i, ctype in enumerate(covariance_types):
                col = cmap.colors[i]      
                sel = self.gridsearch_scores_['cov_type'] == ctype
                results = np.array(self.gridsearch_scores_[sel][["n_components", metric]])
                ax.plot(results[:,0], results[:,1], color=col, label = ctype)

            # mark best
            ax.plot([self.hyperparameters_['n_components']], [self.scores_val_[metric]], color=tudcolors.get("rood"), marker="o")
            ax.annotate(f'{self.scores_val_[metric]:.2f}',
                xy=(self.hyperparameters_['n_components'], self.scores_val_[metric]), 
                horizontalalignment='left',
                verticalalignment='bottom')
            
            ax.set_title(metric)
            ax.set_ylabel("score")
            ax.set_xlabel("n_components")

        axes[0].legend()
        fig.suptitle(f'Grid Search GMM {self.feature_set} Model Selection: {self.riderbike_model}\n normalized: {self.normalize}, power-transfrom: {self.pt_type}')

        if self.save:
        
            if self.riderbike_model is not None:
                filepath = self.riderbike_model + "_"
            else:
                filepath = ""

            filepath = os.path.join(self.paths["dir_out"], filepath+f"{self.feature_set}_{self.OUTPUT_FNAMES["gridsearch-results"]}.png")
            fig.set_size_inches(16.5, 9.5)
            fig.savefig(filepath)

    def plot_marginals(self, X_train, X_test, k_crossval, marginals_2d = True, marginals_1d = True, plot_for_paper=False, 
                       sct_train_style=dict(s=5, color='black'), 
                       sct_test_style=dict(s=5, color=tudcolors.get('roze')),
                       sct_means_style=dict(s=5, color=tudcolors.get('rood')),
                       cond_density_style=dict(linewidth=1, color=tudcolors.get("donkerblauw")),
                       grid_style=None):
        """ Plot the marignal (and conditional) distributions of the fitted model."""

        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before plotting by calling 'fit_optimize()")
        
        n_features = self.gmm_.means_.shape[1]

        if marginals_2d and n_features >= 2:
            fig, ax = self._plot_2d_marginals(X_train, X_test, k_crossval, plot_for_paper, sct_train_style=sct_train_style,
                                              sct_test_style=sct_test_style, sct_means_style=sct_means_style, 
                                              cond_density_style=cond_density_style, grid_style=grid_style)
            #self._plot_2d_marginals(X_train, self.pp_pipeline.transform(self.sample(10000)), k_crossval=10)
        if marginals_1d:
            fig, ax = self._plot_1d_marginals(X_train, X_test, k_crossval)

        return fig, ax
        
    def _plot_1d_marginals(self, X_train, X_test, k_crossval):
        """ Plot the 1d marignal (and conditional) distributions of the fitted model."""

        # create gridspec 
        n_features = self.gmm_.means_.shape[1]
        n_plotsperrow_max = 8
        n_columns = min(n_features, n_plotsperrow_max)
        n_rows = int(np.ceil(n_features/n_plotsperrow_max))

        fig = plt.figure(layout="constrained")
        gs = GridSpec(n_rows, n_columns, figure=fig)
        axes = []

        # rescale data
        X_train_rescaled = self.pp_pipeline.inverse_transform(X_train)
        if not X_test is None:
            X_test_rescaled = self.pp_pipeline.inverse_transform(X_test)

        for i in range(n_features):
            ax = fig.add_subplot(gs[i//n_columns, i%n_columns])
            axes.append(ax)

            # grid for surface plot
            xlim = (np.min(X_train[:, i]) - 1, np.max(X_train[:, i]) + 1)

            # accumulate density
            locations, density = self.gmm_.eval_1d_marginal_pdf(xlim, i)   

            # rescale
            locations = self.pp_pipeline.inverse_transform(locations[:,np.newaxis], sparse_column_indices=[i,]).flatten()

            # plot
            ax.hist(X_train_rescaled[:, i], color='black', density=True, bins=100, label='training samples')
            ax.plot(locations, density, color=tudcolors.get('blauw'), label='model distribution')
            if not X_test is None:
                ax.scatter(X_test_rescaled[:, i], self.gmm_.eval_1d_marginal_pdf_samples(X_test[:, i], i)[1], s=5, color=tudcolors.get('roze'), zorder=100, label='test samples')

            #set plot limits
            location_limits = [np.nanmin(locations), np.nanmax(locations)] 
            
            if not X_test is None:
                minplot = min((np.nanmin(X_train_rescaled[:, i]) - 1, np.nanmin(X_test_rescaled[:, i]) - 1))
                maxplot = max((np.nanmax(X_train_rescaled[:, i]) + 1, np.nanmax(X_test_rescaled[:, i]) + 1))
            else:
                minplot = np.nanmin(X_train_rescaled[:, i]) - 1
                maxplot = np.nanmax(X_train_rescaled[:, i]) + 1

            minplot = max(minplot, location_limits[0])
            maxplot = min(maxplot, location_limits[1])

            ax.set_xlim(minplot, maxplot)
            ax.set_xlabel(self.features[i])

        axes[0].set_ylabel("density")
        axes[-1].legend()

        # title
        valstr = f"{k_crossval:<2}-CROSSVAL"
        teststr = f"TEST"
        nparams = self.gmm_._n_parameters()
        fig.suptitle((f"Best {self.feature_set} Gaussian-Mixture-Model: {self.riderbike_model}\n"
                f"n_components = {self.hyperparameters_['n_components']}, preprocessing: {self.pp_pipeline.method_str}, {self.hyperparameters_['cov_type']}, {nparams} params\n"
                f"      {valstr:<11}  {teststr:<5}\n"
                f"BIC:  {self.scores_val_['BIC']:<11.2f}  {self.scores_test_['BIC']:<5.2f}\n"
                f"AIC:  {self.scores_val_['AIC']:<11.2f}  {self.scores_test_['AIC']:<5.2f}\n"
                f"NLL:  {self.scores_val_['NLL']:<11.2f}  {self.scores_test_['NLL']:<5.2f}\n"))
        
        if self.save:
        
            if self.riderbike_model is not None:
                filepath = self.riderbike_model + "_"
            else:
                filepath = ""

            filepath = os.path.join(self.paths["dir_out"], filepath+f"{self.feature_set}_1d-{self.OUTPUT_FNAMES["marginal-distributions"]}.png")
            fig.set_size_inches(max(3.5*n_columns, 8), max(3.5 * n_rows, 6.5))
            fig.savefig(filepath)

        return fig, ax


    def _plot_2d_marginals(self, X_train=None, X_test=None, k_crossval=None, plot_for_paper=False,
                           sct_train_style=dict(s=5, color='black'), 
                           sct_test_style=dict(s=5, color=tudcolors.get('roze')),
                           sct_means_style=dict(s=5, color=tudcolors.get('rood')),
                           cond_density_style=dict(linewidth=1, color=tudcolors.get("donkerblauw")),
                           grid_style=None):
        """ Plot the 2d marignal (and conditional) distributions of the fitted model."""

        # get feature index pairs
        n_features = self.gmm_.means_.shape[1]
        feature_pairs = np.array(np.triu_indices(n_features, k=1))
        feature_pairs = np.array(feature_pairs).T.tolist()
        n_pairs = len(feature_pairs)

        # create gridspec 
        n_plotsperrow_max = 5
        n_columns = min(n_pairs, n_plotsperrow_max)
        n_rows = int(np.ceil(n_pairs/n_plotsperrow_max))

        fig = plt.figure(layout="constrained")
        gs = GridSpec(n_rows, n_columns, figure=fig)
        axes = []

        # rescale data
        if X_train is not None:
            X_train_rescaled = self.pp_pipeline.inverse_transform(X_train)
        if X_test is not None:
            X_test_rescaled = self.pp_pipeline.inverse_transform(X_test)

        means_rescaled = self.pp_pipeline.inverse_transform(self.gmm_.means_)
        speeds_cond = np.linspace(1.5,5.5,50)
        if isinstance(self.gmm_, ConditionalGaussianMixture):
            means_speed_cond = self.get_component_means(speeds_cond)

        for idx, (i, j) in enumerate(feature_pairs):
            ax = fig.add_subplot(gs[idx//n_columns, idx%n_columns])
            axes.append(ax)

            #grid
            if isinstance(grid_style, dict):
                ax.grid(**grid_style)

            # grid for surface plot
            xlim = [np.min(X_train[:, i]) - 1, np.max(X_train[:, i]) + 1]
            ylim = [np.min(X_train[:, j]) - 1, np.max(X_train[:, j]) + 1]

            #pattern=r"p\d_(.{1,5})"
            #for klim, k in zip([xlim, ylim], [i,j]):
            #    if re.findall(pattern, self.features[k])[0] in ["real"]:
            #        klim[1] = 0
            #    if re.findall(pattern, self.features[k])[0] in ["imag", "mag"]:
            #        klim[0] = 0
            #    if re.findall(pattern, self.features[k])[0] in ["ang"]:
            #        klim[0] = np.pi/2

            # accumulate density
            locations, density = self.gmm_.eval_2d_marginal_pdf(xlim, ylim, i, j)
            N = int(np.sqrt(density.size))

            # rescale
            locations = self.pp_pipeline.inverse_transform(locations, sparse_column_indices=[i,j])
            locations = locations.reshape(N, N, 2)
            density = density.reshape(N, N)

            # plot distribution
            ax.contourf(locations[:,:,0], locations[:,:,1], density, levels=30, 
                        cmap=tudcolors.colormap(name="turkoois"), alpha=0.8)
            ax.contour(locations[:,:,0], locations[:,:,1], density, levels=30, colors='gray', linewidths=0.2)
            
            # plot samples
            if X_train is not None:
                ax.scatter(X_train_rescaled[:, i], X_train_rescaled[:, j], **sct_train_style, zorder=1000)
            if X_test is not None:
                ax.scatter(X_test_rescaled[:, i], X_test_rescaled[:, j], **sct_test_style, zorder=2000)
            
            # plot means
            for k in range(means_rescaled.shape[0]):
                ax.scatter(means_rescaled[k, i], means_rescaled[k, j], **sct_means_style, zorder=3000)
                ax.annotate(str(k), xy=(means_rescaled[k, i], means_rescaled[k, j]+(np.mean(means_rescaled[:, j])*np.sign(means_rescaled[k, j])*0.1)), color=sct_means_style['color'], zorder=3000)

            #set plot limits
            location_limits = [(np.nanmin(locations[:, :, 0]), np.nanmax(locations[:, :, 0])), 
                               (np.nanmin(locations[:, :, 1]), np.nanmax(locations[:, :, 1]))]

            for k, func, loc_lim in zip([i,j],[ax.set_xlim, ax.set_ylim], location_limits):
                if not X_test is None:
                    minplot = min((np.nanmin(X_train_rescaled[:, k]) - 1, np.nanmin(X_test_rescaled[:, k]) - 1))
                    maxplot = max((np.nanmax(X_train_rescaled[:, k]) + 1, np.nanmax(X_test_rescaled[:, k]) + 1))
                else:
                    minplot = np.nanmin(X_train_rescaled[:, k]) - 1
                    maxplot = np.nanmax(X_train_rescaled[:, k]) + 1

                minplot = max(minplot, loc_lim[0])
                maxplot = min(maxplot, loc_lim[1])

                if plot_for_paper:
                    if 'real' in self.features[k]:
                        maxplot = 0
                    if 'imag' in self.features[k]:
                        minplot = 0
                    if 'ang' in self.features[k]:
                        minplot = np.pi/2
                    if 'mag' in self.features[k]:
                        minplot = 0

                func(minplot, maxplot)


            # plot conditional
            features = [f for f in self.features if f != self.feature_cond]
            if isinstance(self.gmm_, ConditionalGaussianMixture) and self.feature_cond in [self.features[i], self.features[j]]:
                if self.features[i] == self.feature_cond:
                    lim = ylim
                    idx_marg = j   
                else:
                    lim = xlim
                    idx_marg = i
                    
                speeds = (np.array([8,11,14])/3.6).reshape((3,1))
                speeds_scaled = self.pp_pipeline.transform(speeds, sparse_column_indices=[self.features.index('v_mean')])
                
                for v, v_scaled in zip(speeds, speeds_scaled):

                    locations, density = self.gmm_.eval_conditional_marginal_pdf(lim, v_scaled, idx_marg)
                    locations = self.pp_pipeline.inverse_transform(locations[:,np.newaxis], sparse_column_indices=[idx_marg])
                    loc_ext = [0, np.min(np.abs(locations)) * np.sign(locations).flatten()[0]]

                    if self.features[i] == self.feature_cond:
                        ax.plot(density+v, locations, **cond_density_style)
                        ax.plot(np.zeros_like(density)+v, locations, linestyle='--', **cond_density_style)
                        
                        # extend density line for illegal values
                        ax.plot([v,v], loc_ext, **cond_density_style)
                    else:
                        ax.plot(locations, density+v,  **cond_density_style)
                        ax.plot(locations, np.zeros_like(density)+v, linestyle='--', **cond_density_style)

                        # extend density line for illegal values
                        ax.plot(loc_ext, [v,v], **cond_density_style)

                #plot speed means
                for k in range(means_rescaled.shape[0]):
                    if self.features[i] == self.feature_cond:
                        ax.plot(speeds_cond, means_speed_cond[k, features.index(self.features[j]), :], color=sct_means_style['color'], linestyle='--', linewidth=cond_density_style['linewidth'])
                    else:
                        ax.plot(means_speed_cond[k, j, :], speeds_cond, color=sct_means_style['color'], linestyle='--', linewidth=cond_density_style['linewidth'])
            
            #axis labels
            if plot_for_paper:
                ax.set_xlabel(self.tex_feature_labels[i], labelpad=0)
                ax.set_ylabel(self.tex_feature_labels[j], labelpad=1)              
                ax.tick_params(axis='x', pad=1)
                ax.tick_params(axis='y', pad=1)
            else:            
                ax.set_xlabel(self.features[i])
                ax.set_ylabel(self.features[j])


        # title
        if not plot_for_paper:
            valstr = f"{k_crossval:<2}-CROSSVAL"
            teststr = f"TEST"
            nparams = self.gmm_._n_parameters()
            fig.suptitle((f"Best {self.feature_set} Gaussian-Mixture-Model: {self.riderbike_model}\n"
                    f"n_components = {self.hyperparameters_['n_components']}, preprocessing: {self.pp_pipeline.method_str}, {self.hyperparameters_['cov_type']}, {nparams} params\n"
                    f"      {valstr:<11}  {teststr:<5}\n"
                    f"BIC:  {self.scores_val_['BIC']:<11.2f}  {self.scores_test_['BIC']:<5.2f}\n"
                    f"AIC:  {self.scores_val_['AIC']:<11.2f}  {self.scores_test_['AIC']:<5.2f}\n"
                    f"NLL:  {self.scores_val_['NLL']:<11.2f}  {self.scores_test_['NLL']:<5.2f}\n"))
        
        if self.save and not plot_for_paper:
            if self.riderbike_model is not None:
                filepath = self.riderbike_model + "_"
            else:
                filepath = ""

            filepath = os.path.join(self.paths["dir_out"], filepath+f"{self.feature_set}_2d-{self.OUTPUT_FNAMES["marginal-distributions"]}.png")
            fig.set_size_inches(16.5, 9.5)
            fig.savefig(filepath)

        return fig, ax


    def export_to_yaml(self):
        """ Export this model as yaml.
        """

        if not self.is_fitted_:
            raise RuntimeError(f"Fit the pole model to data to create parameters that can be saved!")

        # preprocessing pipeline
        preprocessing_pipe = dict(
            power_transform = self.pp_pipeline.power_transform,
            normalize=self.pp_pipeline.normalize, 
            log_transform=self.pp_pipeline.log_transform)
        
        power_transform_params = {}
        log_transform_params = {}

        for trafo in self.pp_pipeline.transformers_:
            if isinstance(trafo, PowerTransformer):
                power_transform_params["lambdas"] = trafo.lambdas_.tolist()
                if self.pp_pipeline.power_transform != 'none' and self.pp_pipeline.normalize:
                    scaler = trafo._scaler
            elif isinstance(trafo, LogTransformer):
                log_transform_params['a'] = trafo.a_.tolist()
                log_transform_params['sign'] = trafo.sign_.tolist()
                log_transform_params['log_transform_features'] = self.pp_pipeline.log_transform_features_.tolist()
            elif isinstance(trafo, StandardScaler):
                scaler = scaler

        if self.pp_pipeline.normalize:
            standard_scaler_params = dict(
                mean = scaler.mean_.tolist(),
                scale = scaler.scale_.tolist(),
                n_samples_seen = int(scaler.n_samples_seen_))
        else:
            standard_scaler_params = {}
        
        preprocessing_pipe['power_transform_params'] = power_transform_params
        preprocessing_pipe['standard_scaler_params'] = standard_scaler_params
        preprocessing_pipe['log_transform_params'] = log_transform_params
        

        # gaussian mixture model 
        gmm = dict(
            means = self.gmm_.means_.tolist(),
            covariances = self.gmm_.get_full_covariancematrix().tolist(), 
            weights = self.gmm_.weights_.tolist(),
            scores_val = self.scores_val_,
            scores_test = {k: float(v) for k, v in self.scores_test_.items()},
            n_samples_test = self.n_samples_test_,
            n_samples_train = self.n_samples_train_,
            n_features = self.gmm_.means_.shape[1],
            n_components = self.gmm_.means_.shape[0],
            k_crossval = self.k_crossval_,
            covariance_type = self.gmm_.covariance_type
        )

        presets = dict(
            feature_set = self.feature_set,
            features = self.features,
            gridsearch_selection_metric=self.gs_sel_metric, 
            n_gmm_inits=self._n_gmm_inits,
            riderbike_model=self.riderbike_model,
        )
        metadata = dict(
            data_created=str(datetime.now())
        )

        data = dict(
            presets = presets, 
            gmm_data = gmm,
            preprocessing_pipeline=preprocessing_pipe,
            metadata = metadata
        )

        #save to yaml
        if self.riderbike_model is not None:
            filepath = self.riderbike_model + "_"
        else:
            filepath = ""

        filepath = os.path.join(self.paths["dir_out"], filepath+f"{self.feature_set}_{self.OUTPUT_FNAMES["model-export"]}.yaml")

        with open(filepath, "w") as f:
            yaml.dump(data, f) 

    def import_from_yaml(filepath, save=False, dir_out="", random_state=None):
        """ Create a PoleModel object from a yaml image as created by PoleModel.export_to_yaml().

        Results in a fitted PoleModel object that can be used to sample new poles. 

        Parameters
        ----------
        filepath : str
            File path to the yaml image.
        save : bool, optional
            Configure to pole model to save any output it creates to file.
        dir_out : None, optional
            Configure the output director of this pole model. 
        random_state : None, optional
            If an integer, use the integer as fixed random seed for the generation of random numbers. 

        Returns:
        --------
        pm : PoleModel
            A fitted pole-model object
        """

        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        # PoleModel object
        paths = dict(dir_out=dir_out)
        pm = PoleModel(paths, 
                       gridsearch_selection_metric= data["presets"]["gridsearch_selection_metric"],
                       normalization=data["preprocessing_pipeline"]["normalize"],
                       power_transformation=data["preprocessing_pipeline"]["power_transform"],
                       feature_set=data['presets']['feature_set'],
                       n_gmm_inits=data['presets']['n_gmm_inits'], 
                       riderbike_model=data['presets']['riderbike_model'], 
                       save=save,
                       from_data=False,
                       random_state=random_state)
        
        # Configure Preprocessing Pipeline
        kwargs_pp = data["preprocessing_pipeline"]
        kwargs_pp['save'] = save
        kwargs_pp['dir_out'] = dir_out
        kwargs_pp['tag'] = pm.riderbike_model

        pm.pp_pipeline = PreprocessingPipeline.from_parameters(
            pm.feature_set,
            pm.features,
            **kwargs_pp
        )

        # Configure GMM
        pm.gmm_, _ = pm._init_gmm_from_params(data["gmm_data"]["means"], 
                                              data["gmm_data"]["covariances"],
                                              data["gmm_data"]["weights"],
                                              random_state)
        
        pm.scores_test_ = data["gmm_data"]["scores_test"]
        pm.scores_val_ = pd.DataFrame(data["gmm_data"]["scores_val"], index=[0])
        pm.k_crossval_ = data["gmm_data"]["k_crossval"]
        pm.n_features_ = data["gmm_data"]["n_features"]
        pm.n_samples_train_ = data["gmm_data"]["n_samples_train"]
        pm.n_samples_test_ = data["gmm_data"]["n_samples_test"]
        pm.hyperparameters_ = pd.DataFrame(dict(cov_type=data["gmm_data"]["covariance_type"],
                                   n_components=data["gmm_data"]["n_components"]), index=[0])
        pm.is_fitted_ = True
        return pm


class PoleModelTest():

    REQUIRED_PATHS = [
        "filepath_polemodel",
        "filepath_partition",
        "dir_data_postprocessed",
        "dir_out"
    ]
    OPTIONAL_PATHS = dict(
        plot_id_result = ["filepath_identification_result"])

    SUBDIRS = {"output-dirname": os.path.join("pole-modeling", "test_predicted-variance")}

    BIKEMODELS = {
        'balancingrider': FixedSpeedBalancingRiderBicycle,
        'planarpoint': FixedSpeedPlanarPointBicycle
    }

    def __init__(self, 
                 path_manager,
                 riderbike_model_id,
                 pole_model_id,
                 n_predictions=1000,
                 save=False,
                 close_figs=False,
                 bikemodel='balancingrider',
                 outliers=[], 
                 plot_id_result=True):
        
        # model ids
        self.riderbike_model_id = riderbike_model_id
        self.pole_model_id = pole_model_id

        #plotting
        self.plot_id_result = plot_id_result

        #paths
        self.paths = self._get_paths(path_manager)

        # bikemodel
        if bikemodel in self.BIKEMODELS.keys():
            self.bikemodel = bikemodel
        else:
            raise ValueError (f"Unknown bikemodel {bikemodel}! Choose any of {list(self.BIKEMODELS.keys())}")
        
        # poles
        if self.bikemodel == "balancingrider":
            self.n_poles = 5
        else:
            self.n_poles = 1
        self.id_pole_keys = [f"pole{i}" for i in range(self.n_poles)]

        # identification results
        if self.plot_id_result:
            self.identifications = pd.read_csv(self.paths["filepath_identification_result"], sep=";")
        self.responsetimes_cs = pd.read_csv(path_manager.getfilepath_reactiontimes(responsetime_definition='csteer'), sep=",")
        self.responsetimes_rf = pd.read_csv(path_manager.getfilepath_reactiontimes(responsetime_definition='yecomp'), sep=",")

        # prediction results
        self.n_predictions = n_predictions
        self.pole_model = PoleModel.import_from_yaml(self.paths["filepath_polemodel"], random_state=42)
        self.tag = f"{self.pole_model.riderbike_model}_{self.pole_model.feature_set}"

        # validation data
        self.partition = read_yaml(self.paths["filepath_partition"])
        self.dataman = RCIDDataManager(self.paths["dir_data_postprocessed"])
        self.outliers = outliers

        # data calibration (for target locations)
        calibration_file = "calibration.yaml"
        self.calibration = read_yaml(os.path.join(self.paths["dir_data_postprocessed"], calibration_file))

        #output
        self.save = save
        self.close_figs = close_figs

        self.significance_level = 0.05
        self.test_results_all_scenes = []
        self.sample_evaluation_results = {}
        self.groundtruth_results_all_scenes = {}
        self.scene_metadata =  {'sample_id': [],
                                'commanded_speed_km/h': [], 
                                'command_frequency_center_Hz':[],
                                'participant': []}

    def _get_paths(self, path_manager):
        """ Get required paths from the path manager. """

        paths = {}
        paths['filepath_polemodel'] = path_manager.getfilepath_polemodel(self.riderbike_model_id, self.pole_model_id)
        paths['filepath_partition'] = path_manager.getfilepath_partition()
        paths['filepath_identification_result'] = path_manager.getfilepath_bestidresult(self.riderbike_model_id)
        paths['dir_identification_result'] = path_manager.getdir_id(self.riderbike_model_id)
        paths['dir_data_postprocessed'] = path_manager.getdir_data_processed() 
        paths['dir_out'] = path_manager.getdir_pm_testvariance(self.riderbike_model_id, pole_model_id=self.pole_model_id, new=True)


        return paths
        
    def _load_identification_result(self, part, split_id, guess):
        if self.plot_id_result:
            # find solution directory
            basedir = self.paths['dir_identification_result']
            for d in os.listdir(basedir):
                if part in d:
                    solution_dir = os.path.join(basedir, d, "solutions")
                    plot_dir = os.path.join(basedir, d, "plots")

            # find file
            pattern = r"trajxy_(\d{1,2})-(\d{1,2}).png"
            for f in os.listdir(plot_dir):
                if re.findall(pattern, f):
                    matches = re.findall(pattern, f)
                    if int(matches[0][0])==split_id:
                        solution_filepath = os.path.join(solution_dir, f"sol_{split_id}-{matches[0][1]}.pkl")
                        break

            #load plotted solution
            with open(solution_filepath, 'rb') as f:
                data = pkl.load(f)

            x = np.array(data['solution']['states']['p_x'])
            y = np.array(data['solution']['states']['p_y'])
            xy_plotted = np.c_[x,y]

            #load resultfile solution
            solution_filepath = os.path.join(solution_dir, f"sol_{split_id}-{guess}.pkl")
            with open(solution_filepath, 'rb') as f:
                data = pkl.load(f)

            x = np.array(data['solution']['states']['p_x'])
            y = np.array(data['solution']['states']['p_y'])
            xy_resultfile = np.c_[x,y]

            if self.bikemodel == 'balancingrider':
                s0sim = [
                    np.array(data['solution']['states']['p_x'][0]),
                    np.array(data['solution']['states']['p_y'][0]),
                    np.array(data['solution']['states']['psi'][0]),
                    0,
                    np.array(data['solution']['states']['delta'][0]),
                    np.array(data['solution']['states']['phi'][0]),
                    np.array(data['solution']['states']['deltadot'][0]),
                    np.array(data['solution']['states']['phidot'][0])]
            else:
                s0sim = [
                    np.array(data['solution']['states']['p_x'][0]),
                    np.array(data['solution']['states']['p_y'][0]),
                    np.array(data['solution']['states']['psi'][0]),
                    0,]
        
            s0sim = np.array(s0sim)

            return xy_plotted, xy_resultfile, s0sim
        else:
            return None, None, None
    

    def _shift_command(self, p_x_c, p_y_c, i_shift, T=4):
        """ Shift the command signal by the response time tau. Pad the beginning and the end
        so that the full signal is always 4 s (0.5 s warmup + 3.5 s maximum measured signal length)
        """

        # full test singal time
        N = int(round(T/T_S))

        # measured test signal time
        N_cmd = p_x_c.size
        
        # x-component
        p_x_c_shift = np.zeros(N)
        p_x_c_shift[:i_shift] = p_x_c[0]
        p_x_c_shift[i_shift:min(N_cmd+i_shift, N)] = p_x_c[:min(N_cmd, N-i_shift)]
        p_x_c_shift[N_cmd:] = p_x_c[-1]

        # y-component
        p_y_c_shift = np.zeros(N)
        p_y_c_shift[:i_shift] = p_y_c[0]
        p_y_c_shift[i_shift:min(N_cmd+i_shift, N)] = p_y_c[:min(N_cmd, N-i_shift)]
        p_y_c_shift[N_cmd:] = p_y_c[-1]

        return p_x_c_shift, p_y_c_shift, N
    
    def _load_responsetime(self, f):
        
        t_cs = self.responsetimes_cs[self.responsetimes_cs["sample_id"] == f]["response_time_s"].iloc[0]
        t_rf = self.responsetimes_rf[self.responsetimes_rf["sample_id"] == f]["response_time_s"].iloc[0]

        i_cs = int(round(t_cs/T_S))
        i_rf = int(round(t_rf/T_S))

        if self.bikemodel == 'balancingrider':
            i_shift = i_cs
        elif self.bikemodel == 'planarpoint':
            i_shift = i_rf
        else:
            raise ValueError("Unknown bike model!")

        return t_cs, i_cs, i_rf, i_shift
    
    def _get_target_locs(self, f):
        """ Get the target locations for the scene f
        """
        part = str(f[1:4])
        target_locs = None
        for day in self.calibration.values():
            if part in day["participants"]:
                target_locs = np.array(day["target_locations"])
        if target_locs is None:
            raise ValueError(f"Can't find target_locations for participant '{part}' in the calibration file")  
        
        return target_locs

    def _get_identification_result(self, f, trk, i0, p_x_c_shift, p_y_c_shift):
        """ Return the identification result and a simulation of the identified poles of file f."""
        id = self.identifications[f==self.identifications['sample_id']].iloc[0]
        _, xy_idresult, s0 = self._load_identification_result(str(id['participant']), id['index_x'], id['guess'])
        poles_identification = self.identifications[f==self.identifications['sample_id']][self.id_pole_keys].iloc[0].apply(lambda p: complex(p)).to_list()

        # target loations
        target_locs = self._get_target_locs(f)

        # speed
        s0[3] = trk["v"][i0]
        if len(p_x_c_shift)-len(trk["v"]) > 0:
            v = np.pad(trk["v"], (0, len(p_x_c_shift)-len(trk["v"])), mode='edge')
        else:
            v = trk["v"][:len(p_x_c_shift)]
        
        # simulation length
        i_max = len(trk["v"][i0:])
            
        #i_maxsim = len(trk["v"])
        test_sim = FixedInputZigZagTest(
                s0, 
                p_x_c_shift[i0:], 
                p_y_c_shift[i0:], 
                v[i0:],
                target_locs, 
                cyclist_name=f"id_result", 
                bike_class=self.BIKEMODELS[self.bikemodel],
                poles = poles_identification,
                animate=False, 
                verbose=False)
            
        test_sim.run()
        xy_idsim = test_sim.bike.traj[[0,1],:i_max].T

        return poles_identification, xy_idsim, xy_idresult


    def _collect_scene_metadata(self, trk):
        self.scene_metadata['sample_id'].append(trk.track_id)
        self.scene_metadata['commanded_speed_km/h'].append(trk.metadata['v_cmd'])
        self.scene_metadata['command_frequency_center_Hz'].append(trk.metadata['f_cmd'])
        self.scene_metadata['participant'].append(trk.metadata['participant_id'])


    def run_scene(self, i_scene, scene, n_val_scenes=None, plot_for_paper=False, distance=20, sample_evaluator_kwargs={}):

        if self.plot_id_result:
            if np.sum(scene==self.identifications['sample_id'])==0:
                raise ValueError(f"Can't find scene '{scene}' in the identification results!")
            
            step_id = self.identifications[self.identifications['sample_id']==scene]['Unnamed: 0'].iloc[0]
            if i_scene is None or n_val_scenes is None:
                print(f"    Testing step {step_id}: {scene}")
            else:
                print(f"    Testing step {step_id}: {scene} (scene {i_scene}/{n_val_scenes})")
        else:
            print(f"    Testing scene {scene}")

        # load scene data
        trk = self.dataman.load_split(scene, subset='steps')
        self._collect_scene_metadata(trk)

        # compensate responsetime
        T = distance / np.mean(trk['v'])
        t_cs, i_cs, i_rf, i_shift = self._load_responsetime(scene)
        i_step = np.argwhere(np.abs(np.diff(trk["p_y_c"]))>0).flatten()[0]
        p_x_c_shift, p_y_c_shift, N = self._shift_command(trk["p_x_c"], trk["p_y_c"], i_shift, T=T)
        
        # adjust speed: the "measured" speed in trk does not correspond well to the position trajetories. For validation, 
        # we derive the speed directly from the position trajectories. 
        v_adj = np.sqrt((trk['p_x'][:-2]-trk['p_x'][2:])**2 + (trk['p_y'][:-2]-trk['p_y'][2:])**2)/(2*T_S)
        if N - len(v_adj) > 0:
            v_adj = np.pad(v_adj, (0, N - len(v_adj)), mode='edge')
        else:
            v_adj = v_adj[:N]

        # scene initial conditions
        target_locs = self._get_target_locs(scene)
        s0 = trk.get_state(i_shift, model=self.bikemodel)
        i_max = len(v_adj[i_shift:])


        # convert pole features to poles
        pole_predictions, labels = self.pole_model.sample_poles(self.n_predictions, np.mean(trk['v']))

        # create sample evaluator
        sample_evaluator_kwargs['plot_for_paper'] = plot_for_paper
        sample_evaluator_kwargs['draw_labels'] = True
        sample_evaluator_kwargs['tag'] = self.tag
        sample_eval = SampleEvaluator(trk["p_x"], trk["p_y"], trk["psi"], 
                                        trk["p_x_c"], trk["p_y_c"],
                                        scene, 
                                        i_cs, i_rf, i_step, i_shift,
                                        np.mean(trk["v"][i_shift:]),
                                        **sample_evaluator_kwargs)

        #identification result
        if self.plot_id_result:
            poles_id, xy_idsim, xy_idresult = self._get_identification_result(scene, trk, i_shift, p_x_c_shift, p_y_c_shift)
            poles_id = np.sort(poles_id)
            if self.bikemodel=="balancingrider":
                print((f"        Identified poles: "
                        f"p_0={poles_id[0]:.2f}, "
                        f"p_1={poles_id[1]:.2f}, "
                        f"p_2={poles_id[3]:.2f}, "))
            else:
                print((f"        Identified poles: "
                        f"p_0={poles_id[0]:.2f}"))
            sample_eval.add_identification(xy_idsim, xy_idresult)

        # simulate predictions
        for i in range(self.n_predictions):
            
            # find next stable pole prediction
            poles = pole_predictions[i,:]

            # simulate ith prediction
            if self.bikemodel=="balancingrider":
                print((f"        Simulating pole predicition {i}: "
                    f"p_0={poles[0]:.2f}, "
                    f"p_1={poles[1]:.2f}, "
                    f"p_2={poles[2]:.2f}, "))
            else:
                print((f"        Simulating pole predicition {i}: "
                    f"p_0={poles[0]:.2f}, "))

            pred_sim = FixedInputZigZagTest(
                s0, 
                p_x_c_shift[i_shift:], 
                p_y_c_shift[i_shift:], 
                v_adj[i_shift-1:], 
                target_locs, 
                cyclist_name=f"pred_{i}", 
                bike_class=self.BIKEMODELS[self.bikemodel],
                poles = poles,
                animate=False, 
                verbose=False)
            pred_sim.run()

            # evaluate ith prediction
            sample_eval.add_pred_sample(i, pred_sim.bike.traj[0,:i_max], pred_sim.bike.traj[1,:i_max], pred_sim.bike.traj[2,:i_max], labels[i])

        # scene level evaluation
        sample_eval.eval_groundtruth_metrics()
        sample_eval.plot_histograms()
        
        return sample_eval
    

    def run(self, n_predictions=None, n_val_scenes=None):
        """Run validation.

        Loop through all test samples in the dataset and simulate predictions. Evaluate the results
        """

        if n_predictions is None:
            n_predictions = self.n_predictions
        else:
            self.n_predictions = n_predictions

        scenes = [f for f in self.partition['test'] if f not in self.outliers]
        n_all_scenes = len(scenes)

        if n_val_scenes is None:
            n_val_scenes = len(scenes)
        elif not (0 < n_val_scenes < len(scenes)):
            n_val_scenes = len(scenes)
        self.n_val_scenes = n_val_scenes

        scenes = scenes[:self.n_val_scenes]

        print("Starting validation.")
        print(f"Predicting {n_predictions} samples for {n_val_scenes}/{n_all_scenes} scenes. Ignoring {len(self.partition['test']) - n_all_scenes} outliers from identifiation.")
        files = []

        for i_scene, step_id in enumerate(scenes):

            step_id = step_id.split('.')[0]
            files.append(step_id)

            # evaluate scene
            sample_eval = self.run_scene(i_scene, step_id, n_val_scenes=n_val_scenes)
            
            # store scene and sample evaluation results
            self.test_results_all_scenes.append(sample_eval.eval_total_metrics())
            self.sample_evaluation_results = {
                k: self.sample_evaluation_results.get(k, []) + sample_eval.prediction_sample_metrics.get(k, [])
                for k in sample_eval.prediction_sample_metrics if k not in ['sample_index']
            }

            self.sample_evaluation_results['files'] = files
            
            self.groundtruth_results_all_scenes = {
                k: self.groundtruth_results_all_scenes.get(k, []) + sample_eval.groundtruth_metrics.get(k, [])
                for k in sample_eval.groundtruth_metrics
            }
            self.groundtruth_results_all_scenes['files'] = files

            if self.save:
                sample_eval.savefig(self.paths["dir_out"], self.tag)
            if self.close_figs:
                plt.close(sample_eval.ax_xy.get_figure())

        self._aggregate_trajectron_NLL(files)
        self.test_results_all_scenes = pd.DataFrame(self.test_results_all_scenes)
        
        self.eval_session()
        plt.show(block=True)


    def _aggregate_trajectron_NLL(self, files):

        NLL = []
        for result in self.test_results_all_scenes:
            NLL.append(result['T-NLL'])
            result['T-NLL'] = np.mean(result['T-NLL'])

        NLL = np.array(NLL)
        nll_median = np.median(NLL, axis=0)
        nll_75 = np.percentile(NLL, 75, axis=0)
        nll_25 = np.percentile(NLL, 25, axis=0)

        fig, ax = plt.subplots(1,1, layout='constrained')
        ax.set_ylabel('NLL')
        ax.set_xlabel('t-t_cs [s]')
        ax.set_title(f'Trajectron NLL: {self.tag}')

        #plot samples
        t = np.arange(NLL.shape[1])*T_S + 0.5
        for i in range(NLL.shape[0]):
            ax.plot(t, NLL[i,:], linestyle='None', marker='.', color=tudcolors.get('blauw'), alpha=0.25)

        #plot errorbars
        ax.errorbar(t, nll_median, yerr=np.array([np.abs(nll_median-nll_25), np.abs(nll_median-nll_75)]), color=tudcolors.get('rood'))
        ax.set_ylim(min(-15, np.min(nll_25) - 1), max(15, np.max(nll_75) + 1))

        #save plot
        if self.save:
            fig.savefig(os.path.join(self.paths["dir_out"], f"{self.tag}_trajectron-nll.png"))

        #save data
        df = pd.DataFrame(NLL.T, columns=files)
        df.to_csv(os.path.join(self.paths["dir_out"], f"{self.tag}_trajectron-nll.csv"))
    
    def _ks_2sample_test(self, sample1, sample2):

        sample1 = np.array(sample1)
        sample2 = np.array(sample2)

        if (np.sum(np.isfinite(sample1))<2) or (np.sum(np.isfinite(sample2))<2):
            return np.nan, f"KS-test: not enough values", np.nan

        ks_statistic, p_value = ks_2samp(sample1[np.isfinite(sample1)], sample2[np.isfinite(sample2)])

        test_significance = p_value < self.significance_level

        if p_value < 0.001:
            pvalstr = "p < 0.001"
        else:
            pvalstr = f"p = {p_value:.3}"

        if test_significance:

            resultstr = f"KS-test: {pvalstr}, reject (different dist.)"
        else:
            resultstr = f"KS-test: {pvalstr}, accept (same dist.)"

        return p_value, resultstr, test_significance
    
    def _make_scenemetric_histograms(self):
        
        fig_nll, axes_nll = plt.subplots((self.test_results_all_scenes.shape[1]),1, layout='constrained')
        
        for ax_nll, m in zip(axes_nll, self.test_results_all_scenes.keys()):
            if m == 'sample_id':
                continue
            mean = np.mean(self.test_results_all_scenes[m])
            median = np.median(self.test_results_all_scenes[m])

            if "NLL" in m:
                rng = [-10, 50]
            else:
                rng = [0, 1]
            
            counts, _, _ = ax_nll.hist(self.test_results_all_scenes[m], range=rng, bins=150, label="NLL samples")
            ylim = [0, counts.max()+1]
            ax_nll.plot(np.ones_like(ylim)*mean, ylim, color=tudcolors.get("blauw"), linestyle="dashed", label="mean")
            ax_nll.annotate(f" mean={mean:.4f}", xy=(mean, 0.55 * np.diff(ylim)),
                            horizontalalignment='left', verticalalignment='center', color=tudcolors.get("blauw"))
            ax_nll.plot(np.ones_like(ylim)*median, ylim, color=tudcolors.get("blauw"), linestyle="dotted", label="median")
            ax_nll.annotate(f" median={median:4f}", xy=(median, 0.8 * np.diff(ylim)), 
                            horizontalalignment='left', verticalalignment='center', color=tudcolors.get("blauw"))
            ax_nll.set_xlim(rng)
            ax_nll.set_ylim(ylim)
            ax_nll.set_title(m)
            ax_nll.set_ylabel("n")

        fig_nll.suptitle(f"Distribution of scene metrics, model {self.tag}")
        axes_nll[0].legend()

        fig_nll.set_size_inches(18, 10)

        if self.save:
            fig_LDNLLdist_filepath = os.path.join(self.paths["dir_out"], f'{self.tag}_LDNLL-distributions.png')
            fig_nll.savefig(fig_LDNLLdist_filepath)

    def eval_session(self):

        self._make_scenemetric_histograms()

        #later deviation distribution plot        
        fig, ax = plt.subplots(1,1)
        fig.suptitle(f"Predictive check, model {self.tag}: Lateral deviation (LD) distributions")
        fig.set_size_inches(10,5)
        ax.set_ylabel('Lateral Deviation (LD) [m]')
        ax.set_xlabel('time t [s]')
        for i, ld_pred in enumerate(self.sample_evaluation_results['LD']):
            i_cs = self.sample_evaluation_results['j_cs'][i]
            t = (np.arange(len(ld_pred)) - i_cs) * T_S
            ax.plot(t, ld_pred, color=tudcolors.get('donkergroen'), label='measurements', alpha=0.1)
        for i, ld_gt in enumerate(self.groundtruth_results_all_scenes['LD']):
            i_cs = self.groundtruth_results_all_scenes['i_cs'][i]
            t = (np.arange(len(ld_gt)) - i_cs) * T_S
            ax.plot(t, ld_gt, color=tudcolors.get('rood'), label='measurements')

        # create result dataframes
        df_pred_allscenes = self.test_results_all_scenes
        df_gt_allscenes = pd.DataFrame(self.groundtruth_results_all_scenes)
        df_scene_metadata = pd.DataFrame(self.scene_metadata)

        # create result summary dataframes
        def n_inf(x):
            return np.sum(np.isinf(x))
        def n_nan(x):
            return np.sum(np.isnan(x))
        def q1(x):
            return np.nanpercentile(x, 25)
        def q3(x):
            return np.nanpercentile(x, 75)
        def make_summary(df, tag, filter_column=None, filter_value=None):
            metrics = list(df.keys())
            metrics.remove("sample_id")
            df = pd.merge(df_scene_metadata, df, on='sample_id')
            if filter_column is not None:
                df = df[df[filter_column]==filter_value]
            # make summary dataframe
            summary={}
            ops = dict(mean=np.nanmean, var=np.nanvar, median=np.nanmedian, q1=q1, q3=q3, min=np.min, max=np.max, n=len, n_inf=n_inf, n_nan=n_nan)
            for opname, op in ops.items():
                summary[f'{opname}_{tag}'] = []
                for m in metrics:
                    summary[f'{opname}_{tag}'].append(op(df[m]))
            return pd.DataFrame(summary, index=list(metrics))
        
        df_summary_all = make_summary(df_pred_allscenes, 'all')

        summaries_vcmd = [make_summary(df_pred_allscenes, tag=f'{vcmd:.1f}km/h', filter_column='commanded_speed_km/h', filter_value=vcmd) 
                         for vcmd in np.unique(self.scene_metadata['commanded_speed_km/h'])]
        df_summary_vcmd = pd.concat(summaries_vcmd, axis=1)

        summaries_fcmd = [make_summary(df_pred_allscenes, tag=f'{fcmd:.1f}Hz', filter_column='command_frequency_center_Hz', filter_value=fcmd) 
                         for fcmd in np.unique(self.scene_metadata['command_frequency_center_Hz'])]
        df_summary_fcmd = pd.concat(summaries_fcmd, axis=1)

        summaries_part = [make_summary(df_pred_allscenes, tag=part, filter_column='participant', filter_value=part) 
                         for part in np.unique(self.scene_metadata['participant'])]
        df_summary_part = pd.concat(summaries_part, axis=1)

        #print results
        print("Finished validation!")
        print(f"Predicted variance test summaries, model {self.tag}, n_predictions={self.n_predictions}, n_scenes={self.n_val_scenes}")
        print(df_summary_all)
        print(df_summary_vcmd)
        print(df_summary_fcmd)
        print(df_summary_part)

        # save_results
        if self.save:
            fig_LDdist_file_path = os.path.join(self.paths["dir_out"], f'{self.tag}_LD-distribution.png')
            fig.savefig(fig_LDdist_file_path)
            testresults_file_path = os.path.join(self.paths["dir_out"], f'{self.tag}_test-results_all-scenes.csv')
            df_pred_allscenes.to_csv(testresults_file_path, sep=';')
            gtresults_file_path = os.path.join(self.paths["dir_out"], f'{self.tag}_gt-results_all-scenes.csv')
            df_gt_allscenes.to_csv(gtresults_file_path, sep=';')

            for df_sum, tag in zip([df_summary_all, df_summary_vcmd, df_summary_fcmd, df_summary_part], 
                                   ['all', 'commanded-speed', 'command-frequency', 'participants']): 
                result_summary_file_path = os.path.join(self.paths["dir_out"], f'{self.tag}_test-result-summary_{tag}.csv')
                df_sum.to_csv(result_summary_file_path, sep=';')

        # plot summary
        self._plot_session_summaries(df_summary_all, df_summary_vcmd, df_summary_fcmd, df_summary_part)


    def _plot_session_summaries(self, df_summary_all, df_summary_vcmd, df_summary_fcmd, df_summary_part):

        n_metrics = df_summary_all.shape[0]
        
        commanded_speeds = np.unique(self.scene_metadata['commanded_speed_km/h'])
        command_frequencies = np.unique(self.scene_metadata['command_frequency_center_Hz'])
        participants = np.unique(self.scene_metadata['participant'])

        colors = ['gray', tudcolors.get('cyaan'), tudcolors.get('donkergroen'), tudcolors.get('blauw')]

        fig, axes = plt.subplots(n_metrics, 1, layout='constrained', sharex=True)

        labels = []
        def plot_box(ax, cat_nr, df, tag, color):
            bxpstats = [dict(
                med=df[f'median_{tag}'],
                q1=df[f'q1_{tag}'],
                q3=df[f'q3_{tag}'],
                whislo=df[f'min_{tag}'],
                whishi=df[f'max_{tag}'])]
            
            drawing = ax.bxp(bxpstats, positions=[cat_nr], showfliers=False, patch_artist=True)
            drawing['boxes'][0].set_facecolor(color)
            drawing['medians'][0].set_color(tudcolors.get('rood'))
            labels.append(tag)

        for i, metric in enumerate(df_summary_all.index):
            axes[i].set_ylabel(metric)
            plot_box(axes[i], 0, df_summary_all.loc[metric], 'all', colors[0])

            j = 1
            for vcmd in commanded_speeds:
                plot_box(axes[i], j, df_summary_vcmd.loc[metric], f'{vcmd:.1f}km/h', colors[1])
                j += 1
            for fcmd in command_frequencies:
                plot_box(axes[i], j, df_summary_fcmd.loc[metric], f'{fcmd:.1f}Hz', colors[2])
                j += 1
            for part in participants:
                plot_box(axes[i], j, df_summary_part.loc[metric], part, colors[3])
                j += 1

        axes[-1].set_xticks(np.arange(j))
        axes[-1].set_xticklabels(labels[:j])
        axes[0].set_title(f'Predicted variance test result summary: {self.tag}')
        fig.set_size_inches(18,15)

        if self.save:
            filepath = os.path.join(self.paths['dir_out'], f'{self.tag}_test-result-summary.png')
            fig.savefig(filepath)


class SampleEvaluator():
    
    def __init__(self, x, y, psi, p_x_c, p_y_c, step_id, i_cs, i_rf, i_step, gt_pred_shift, v_mean, tag="", plot_for_paper=False, ax_xy=None,
                 mgridxloc=[0,2,4,8,10,12,14], mgridyloc=[-2,-1,0,1,2], draw_mgrid=True, markerstyle=dict(markersize=5), linewidth=1, fontsize_annotations=6, draw_nll_eval_points=True,
                 draw_fde_eval_points=True, draw_rising_flank_point = True, draw_countersteer_point=True, draw_tau_values=True, draw_scene_characteristics=True,
                 mgridstyle=dict(color="gray", linewidth=1), mgridannotate=True, mgridannotate_x=True, mgridannotate_y=True, draw_labels=False, t_begin_eval=0.5, t_end_eval=2.0):

        self.step_id = step_id
        self.tag = tag

        # important sample indices
        self.N_gt = len(x)
        N_eval_max = int(round(1.4 / T_S))      # maximum number of evaluation samples common for all scenes. 

        self.i_step = i_step                    #sample number in the groundtruth where the true step lies.
        self.i_cs = i_step + i_cs               #sample number in the groundtruth where the countersteer begins.
        self.i_rf = i_step + i_rf               #sample number in the groundtruth where the rising yaw flank begins.
        self.i_pred_begin = gt_pred_shift       #sample number in the groundtruth where the prediction begins.
        self.i_start_eval = self.i_cs + int(round(t_begin_eval / T_S))
        self.i_end_eval = self.i_cs + int(round(t_end_eval / T_S))   #sample number in the groundtruth until which the sample can be evaluated 

        self.j_step = i_step - gt_pred_shift    #sample number in the prediction where the true step lies.
        self.j_cs = self.i_cs - gt_pred_shift   #sample number in the prediction where the countersteer lies.
        self.j_end_eval = self.i_end_eval - gt_pred_shift #sample number in the prediction until which the sample can be evaluated 
        self.j_start_eval = self.i_start_eval - gt_pred_shift

        self.gt_pred_shift = gt_pred_shift       #sample number by which groundtruth and prediction are shifted

        # groundtruth position and command
        self.x0 = x[self.i_cs]
        self.y0 = y[self.i_cs]

        self.x = x - self.x0
        self.y = y - self.y0

        p_x_c = p_x_c - self.x0
        p_y_c = p_y_c - self.y0
        
        xlim = [np.min(x) - self.x0, np.max(x) - self.x0]
        ylim = [np.min(y) - self.y0, np.max(y) - self.y0]

        # sample rotation for lateral yaw calculation
        psi_c = np.arctan2((p_y_c[self.i_cs+1]), (p_x_c[self.i_cs+1]))
        self.psi_c = psi_c
        self.psi_err = psi_c - psi[self.i_cs+1]
        self.sign_psi_error = np.sign(psi_c - psi[self.i_cs+1])
        if self.sign_psi_error == 0.0:
            self.sign_psi_error = 1.0   #if psi is perfe
        self.rot = np.array([[np.cos(psi_c), -np.sin(psi_c)],
                             [np.sin(psi_c), np.cos(psi_c)]])

        # Lateral deviation test locs.    
        test_times = np.array([1, 1.25, 1.5, 1.75, 2.0])
        self.i_test = np.round(test_times/T_S).astype(int) + self.i_cs
        self.i_test = self.i_test[self.i_test < self.N_gt]
        self.j_test = self.i_test - self.gt_pred_shift
        self.t_test = test_times[:len(self.i_test)]

        self.total_metrics = {
            'sample_id': step_id,
            'mean_ADE_m': None, 
            '1-minADE_m': None, '5-minADE_m': None, '10-minADE_m': None, 
            '1-minFDE_m': None, '5-minFDE_m': None, '10-minFDE_m': None, 
            'LD-NLL': None,         #lateral deviation NLL
            #'fixedN_LD-NLL': None,  #lateral deviation NLL with a common fixed number of samples for all scenes.
            'T-NLL': None}          #Trajectron-style NLL
        
        self.prediction_sample_metrics = {'sample_index':[], 'ADE': [], 'FDE': [], 'p_x': [], 'p_y': [], 'LD':[], 'j_cs':[]}
        self.groundtruth_metrics = {'p_x': [], 'p_y': [], 'LD':[], 'i_cs':[]}

        #plotting
        self.mxloc=mgridxloc
        self.myloc=mgridyloc
        self.mgridstyle=mgridstyle
        self.mgridannotate=mgridannotate
        self.mgridannotate_x = mgridannotate & mgridannotate_x
        self.mgridannotate_y = mgridannotate & mgridannotate_y
        self.draw_mgrid = draw_mgrid
        self.plot_for_paper = plot_for_paper
        self.markerstyle = markerstyle
        self.linewidth = linewidth
        self.fontsize_annotations = fontsize_annotations
        self.draw_nll_eval_points = draw_nll_eval_points
        self.draw_fde_eval_points = draw_fde_eval_points
        self.draw_rising_flank_point = draw_rising_flank_point
        self.draw_countersteer_point = draw_countersteer_point
        self.draw_tau_values = draw_tau_values
        self.draw_scene_characteristics = draw_scene_characteristics
        self.draw_labels = draw_labels
        self._make_xy_plot(xlim, ylim, self.x, self.y, psi, v_mean, ax_xy)
        
        
    def _make_xy_plot(self, xlim, ylim, x, y, psi, v_mean, ax_xy):

        fig = plt.figure(figsize=(10, 8), layout="constrained")
        gs = GridSpec(2,5, height_ratios=(3,1), figure=fig)

        #x-y plot
        if ax_xy is None:
            ax_xy = fig.add_subplot(gs[0,:])
        ax_xy.set_aspect("equal")

        if not self.plot_for_paper:
            ax_xy.set_title(f"Predictive check, sample {self.step_id}, model {self.tag}: v_mean = {v_mean:.1f} m/s")
            ax_xy.set_ylabel("y [m]")
            ax_xy.set_xlabel("x [m]")

        #make m-grid
        if self.draw_mgrid:
            m_plotkwargs = self.mgridstyle
            m_xloc = self.mxloc
            m_yloc = self.myloc
            for xloc in m_xloc:
                m = np.array([[xloc, xloc],[min(m_yloc)-0.1, max(m_yloc)+0.1]])
                m = self.rot @ m
                if xloc!=0:
                    m_plotkwargs['linewidth'] /= 2
                ax_xy.plot(m[0,:], m[1,:], **m_plotkwargs)
                if self.mgridannotate_x:
                    ax_xy.annotate(f"{xloc:.1f}", m[:,1], horizontalalignment='right', color=m_plotkwargs['color'], fontsize=self.fontsize_annotations)
                if xloc!=0:
                    m_plotkwargs['linewidth'] *= 2

            for yloc in m_yloc:
                m = np.array([[min(m_xloc)-0.1, max(xlim[1], np.max(m_xloc))+.1],[yloc, yloc]])
                m = self.rot @ m
                if yloc!=0:
                    m_plotkwargs['linewidth'] /= 2
                ax_xy.plot(m[0,:], m[1,:], **m_plotkwargs)
                if self.mgridannotate_y:
                    m = np.array([[min(m_xloc)-0.2, max(xlim[1], np.max(m_xloc))+.1],[yloc, yloc]])
                    m = self.rot @ m
                    ax_xy.annotate(f"{yloc*self.sign_psi_error:.1f}", m[:,0], horizontalalignment='right', color=m_plotkwargs['color'], verticalalignment='center', fontsize=self.fontsize_annotations)
                if yloc!=0:
                    m_plotkwargs['linewidth'] *= 2
            if self.mgridannotate_y:
                if self.plot_for_paper:
                    lbl = r"LD $[\mathrm{m}]$"
                else:
                    lbl = "LD [m]"
                pos_ylabel = self.rot @ np.array([[-1.1], [np.mean(m_yloc)]])
                ax_xy.annotate(lbl, pos_ylabel.flatten(), rotation=np.rad2deg(self.psi_c)+90, rotation_mode='anchor', color=m_plotkwargs['color'], horizontalalignment='center', verticalalignment='center', fontsize=self.fontsize_annotations)
            

        # plot data
        line_meas = ax_xy.plot(x, y, color=tudcolors.get("rood"), zorder=1000, label='test (measurement)', linewidth=self.linewidth)

        # mark special points
        def make_tstr(subscript=None, val=None, unit=None):
            if self.plot_for_paper:
                if subscript is None:
                    tstr = r'$'
                else:
                    tstr = r'$\tau_\mathrm{'+subscript+r'}'
                    if not val is None:
                         tstr += r'='
                if not val is None:
                    tstr += val 
                if not unit is None:
                    tstr += r'~\mathrm{' + unit + r'}'
                tstr += r"$"
            else:
                if subscript is None:
                    tstr = ''
                else:
                    tstr = f't_{subscript}'
                    if not val is None:
                        tstr += '='
                if not val is None:
                    tstr += val 
                if not unit is None:
                    tstr += unit
            return tstr

        def mark(idx, text, marker="o", fillstyle='full'):
            mstyledict = {k:v for k, v in self.markerstyle.items()}
            mstyledict['color'] = tudcolors.get("rood")
            mstyledict['linestyle'] = 'none'

            t = Affine2D().rotate(psi[idx])
            mstyle = MarkerStyle(marker, fillstyle=fillstyle, transform=t)
            mstyledict['marker'] = mstyle

            pos = np.array([x[idx], y[idx]])
            lbl_offset = np.array([0.2, 0.2])
            l = ax_xy.plot(pos[0], pos[1], zorder=1000, **mstyledict)
            t = ax_xy.annotate(text, pos-lbl_offset, horizontalalignment='left', color=tudcolors.get("rood"), 
                        verticalalignment='top', size=self.fontsize_annotations)

        # mark command
        mark(self.i_step, make_tstr(val="0", unit="s"), fillstyle='none')
        
        # mark countersteer
        if self.draw_countersteer_point:
            if self.i_cs - self.i_step == self.gt_pred_shift:
                fs = 'full'
            else:
                fs = 'none'
            if self.draw_tau_values:
                mark(self.i_cs, make_tstr(subscript="cs", val=f"{(self.i_cs-self.i_step)*T_S:.2f}", unit="s"), fillstyle=fs)
            else:
                mark(self.i_cs, make_tstr(subscript="cs"), fillstyle=fs)

        # mark rising yaw flank
        if self.draw_rising_flank_point:
            if self.i_rf - self.i_step == self.gt_pred_shift:
                fs = 'full'
            else:
                fs = 'none'
            if self.draw_tau_values:
                mark(self.i_rf, make_tstr(subscript="ec", val=f"{(self.i_rf-self.i_step)*T_S:.2f}", unit="s"), fillstyle=fs)
            else:
                mark(self.i_rf, make_tstr(subscript="ec"), fillstyle=fs)

        # mark ADE/FDE evaluation points / range
        if not self.plot_for_paper:
            mark(self.i_start_eval, "", marker=">")
            mark(self.i_end_eval, "", marker="<")

        # mark evaluation points 
        if self.draw_nll_eval_points:
            for t, i in zip(self.t_test, self.i_test):
                mark(i, "", marker="|")

        # hist plots
        if not self.plot_for_paper:
            axes_hist = []
            for i, t in enumerate(self.t_test):

                if i > 0:
                    ax = fig.add_subplot(gs[1,i], sharex=axes_hist[-1])
                    #ax.set_ylim(0, 1)
                else:
                    ax = fig.add_subplot(gs[1,i])
                    ax.set_ylabel("density")
                ax.set_xlabel("lateral position [m]")
                ax.set_title(f"Lateral dist., t={t:.1f} s)")
                ax.grid(visible=True, axis='x')

                axes_hist.append(ax)

            self.axes_hist = axes_hist

        # scene annotations
        if self.draw_scene_characteristics:
            if self.plot_for_paper: 
                text = (
                    r"$\bar{v} = " + f"{v_mean:.1f}" + r"~\mathrm{m}~\mathrm{s}^{-1}$" + "\n"
                    r"$\hat{\psi}(\tau_\mathrm{cs}) - \psi(\tau_\mathrm{cs}) = " + f"{np.rad2deg(self.psi_err):.1f}" + r"~^\circ$" 
                )
            else:
                text = (
                    r"psi_error = " + f"{np.rad2deg(self.psi_err):.1f}" + r" deg" + "\n"
                    r"v_mean= " + f"{v_mean:.1f}" + r" m/s"
                )

            at = AnchoredText(text, loc='lower left', frameon=False, prop=dict(size=self.fontsize_annotations))  
            ax_xy.add_artist(at)
        
        self.ax_xy = ax_xy


    def eval_groundtruth_metrics(self):

        self._store_sample_features(self.x, self.y, 'groundtruth')
        
        #pattern_LD = r"LD-([t,d])(\d{1,4})"
        #for m in self.groundtruth_metrics:
        #    if re.findall(pattern_LD, m):
        #        match = re.findall(pattern_LD, m)[0]
        #        if match[0]=='t':
        #            t_eval = int(match[1])/1000
        #            i_eval = int(t_eval / T_S) + self.i_step_gt
        #        elif match[0]=='d':
        #            i_eval = np.argmin(np.abs(xy[0,:] - float(match[1])))
        #            if float(match[1]) > xy[0,-1]:
        #                i_eval = self.x.size+1
        #        else:
        #            raise RuntimeError(f"Unknown metric {m}!")
        #
        #        if i_eval >= self.x.size:
        #            val = np.nan
        #        else:
        #           val = xy[1,i_eval]
        #        
        #        self.groundtruth_metrics[m].append(val)

        return self.groundtruth_metrics
    
    def _store_sample_features(self, x_pred, y_pred, target):
        """ Store the features of a sample either for GT evaluation or Pred evaluation
        """
        xy = self._rotate_sample(x_pred, y_pred)

        if target == 'prediction':        
            target_dict = self.prediction_sample_metrics
            target_dict['j_cs'].append(self.j_cs)
        elif target == 'groundtruth':
            target_dict = self.groundtruth_metrics
            target_dict['i_cs'].append(self.i_cs)

        else:
            raise ValueError(f"'target' must be either 'groundtruth' or 'prediction'. Instead it was '{target}'")
   
        target_dict['LD'].append(xy[1,:])
        target_dict['p_x'].append(x_pred)
        target_dict['p_y'].append(y_pred)
        

    def _rotate_sample(self, x_pred, y_pred):
        
        xy = np.vstack((np.reshape(x_pred, [1,-1]), np.reshape(y_pred, [1,-1])))
        xy = self.rot.T @ xy
        xy[1,:] *= self.sign_psi_error

        return xy

    def eval_sample_metrics(self, sample_idx, x_pred, y_pred):
        """ Evaluate metrics for a single prediction sample
        """

        self.prediction_sample_metrics["sample_index"] = sample_idx
        
        # evaluate ADE
        N = self.x[self.i_start_eval:].size
        self.prediction_sample_metrics["ADE"].append(np.mean(np.sqrt((self.x[self.i_start_eval:] - x_pred[self.j_start_eval:self.j_start_eval+N])**2 + 
                                                          (self.y[self.i_start_eval:] - y_pred[self.j_start_eval:self.j_start_eval+N])**2)))

        # evaluate FDE
        for m in self.prediction_sample_metrics:
            if m == 'FDE':
                val = np.mean(np.sqrt((self.x[self.i_end_eval] - x_pred[self.j_end_eval])**2 + 
                                      (self.y[self.i_end_eval] - y_pred[self.j_end_eval])**2))
                self.prediction_sample_metrics[m].append(val)

        # store features
        self._store_sample_features(x_pred, y_pred, 'prediction')
         
        #pattern_LD = r"LD-([t,d])(\d{1,4})"
        #for m in self.prediction_sample_metrics:
        #    if re.findall(pattern_LD, m):
        #        match = re.findall(pattern_LD, m)[0]
        #        if match[0]=='t':
        #            t_eval = int(match[1])/1000
        #            i_eval = int(t_eval / T_S) + self.i_step_pred
        #        elif match[0]=='d':
        #            i_eval = np.argmin(np.abs(xy[0,:] - float(match[1])))
        #            if float(match[1]) > xy[0,-1]:
        #                i_eval = self.x.size+1
        #        else:
        #            raise RuntimeError(f"Unknown metric {m}!")
        #
        #        if i_eval >= x_pred.size:
        #            val = np.nan
        #        else:
        #            val = xy[1,i_eval]
        #        
        #        self.prediction_sample_metrics[m].append(val)

    def _eval_trajectron_NLL(self):

        key = 'T-NLL'

        X = np.array(self.prediction_sample_metrics['p_x'])[:,self.j_start_eval:self.j_end_eval]
        Y = np.array(self.prediction_sample_metrics['p_y'])[:,self.j_start_eval:self.j_end_eval]

        x = self.x[self.i_start_eval:self.i_end_eval]
        y = self.y[self.i_start_eval:self.i_end_eval]

        nll = np.zeros(X.shape[1]-1)

        for i in range(1, X.shape[1]):
            data = np.array([X[:,i], Y[:,i]])
            pred_dist_i = gaussian_kde(data)
            nll[i-1] = - pred_dist_i.logpdf([[x[i]],[y[i]]])

        self.total_metrics['T-NLL'] = nll

        fig, ax = plt.subplots(1,1)
        ax.plot(nll)

    def _eval_LDNLL(self):
    
        LD = np.array(self.prediction_sample_metrics['LD'])[:,self.j_test].T
        ld = np.array(self.groundtruth_metrics['LD'])[:, self.i_test].reshape(-1,1)

        pred_dist = gaussian_kde(LD)
        nll = - pred_dist.logpdf(ld)

        self.total_metrics['LD-NLL'] = nll[0]

    def _eval_LDNLLCT(self):

        mask = self.i_test <= self.i_end_eval
        i_test = self.i_test[mask]
        j_test = self.j_test[mask]
    
        LD = np.array(self.prediction_sample_metrics['LD'])[:,j_test].T
        ld = np.array(self.groundtruth_metrics['LD'])[:, i_test].reshape(-1,1)

        pred_dist = gaussian_kde(LD)
        nll = - pred_dist.logpdf(ld)

        self.total_metrics['fixedN_LD-NLL'] = nll[0]

    
    def eval_total_metrics(self):
        """ Evaluate metrics for all predictions.
        """

        # sort sample metrics
        ade = self.prediction_sample_metrics['ADE']
        ade = np.sort(ade)
        fde = self.prediction_sample_metrics['FDE']
        fde = np.sort(fde)

        # calculate total metrics
        pattern_minADE = r"(\d{1,3})-minADE_m"
        pattern_minFDE = r"(\d{1,3})-minFDE_m"

        text = ""
        for m in self.total_metrics:
            unit="m"
            if m == 'sample_id':
                continue
            if m == 'mean_ADE_m':
                self.total_metrics['mean_ADE_m'] = np.nanmean(ade)
            elif re.findall(pattern_minADE, m):
                match = re.findall(pattern_minADE, m)[0]
                k = int(match)
                self.total_metrics[m] = np.nanmean(ade[:k]) 
            elif re.findall(pattern_minFDE, m):
                match = re.findall(pattern_minFDE, m)[0]
                k = int(match)
                self.total_metrics[m] = np.nanmean(fde[:k]) 
            elif m == 'T-NLL':
                self._eval_trajectron_NLL()
                unit = ""
            elif m == 'LD-NLL':
                self._eval_LDNLL()
                unit = ""
            #elif m == 'fixedN_LD-NLL':
            #    self._eval_LDNLLCT()
            #    unit = ""
            if not m == 'T-NLL':
                text += f"{m}: {self.total_metrics[m]:.3f} {unit}\n"

        # add result test to figure
        if not self.plot_for_paper:
            at = AnchoredText(text[:-1], loc='lower right')  
            self.ax_xy.add_artist(at)

        return self.total_metrics

    def add_pred_sample(self, sample_idx, x_pred, y_pred, psi_pred, label=2):

        x_pred = x_pred - self.x0
        y_pred = y_pred - self.y0

        # metrics
        self.eval_sample_metrics(sample_idx, x_pred, y_pred)
        
        # plot trace
        if self.plot_for_paper:
            j0 = self.j_start_eval
            j1 = self.j_end_eval

            lstyle = dict(alpha=0.15, linewidth=self.linewidth*0.5)

            self.ax_xy.plot(x_pred[:j0+1], y_pred[:j0+1], color='black', **lstyle)
            self.ax_xy.plot(x_pred[j0:j1], y_pred[j0:j1], color=tudcolors.get("donkergroen"), **lstyle)
            self.ax_xy.plot(x_pred[j1-1:], y_pred[j1-1:], color='black', **lstyle)
        else:
            if self.draw_labels:
                c = cmap.colors[label]
            else:
                c = tudcolors.get('donkergroen')
            self.ax_xy.plot(x_pred, y_pred, color=cmap.colors[label], alpha=0.15, label='predictions')


        # plot NLL eval points
        mstyle = {k:v for k, v in self.markerstyle.items()}
        mstyle['color'] = tudcolors.get("donkergroen")
        mstyle['alpha'] = 0.25
        
        if self.draw_nll_eval_points:
            for j in self.j_test:
                mstyle['marker'] = MarkerStyle('|', transform=Affine2D().rotate(psi_pred[j]))
                l = self.ax_xy.plot(x_pred[j], y_pred[j], **mstyle)

        # plot FED eval point
        mstyle['markeredgecolor'] = 'none'
        if self.draw_fde_eval_points:
            mstyle['marker'] = MarkerStyle('<', transform=Affine2D().rotate(psi_pred[self.j_end_eval]))
            l = self.ax_xy.plot(x_pred[self.j_end_eval], y_pred[self.j_end_eval], **mstyle)
    
        if sample_idx == 0 and not self.plot_for_paper:
            self.ax_xy.legend(loc="upper left")


    def add_identification(self, xy_sim, xy_id=None):

        x_pred = xy_sim[:,0] - self.x0
        y_pred = xy_sim[:,1] - self.y0

        #plot trace
        line_sim = self.ax_xy.plot(x_pred, y_pred, color=tudcolors.get("rood"), linestyle = "dashed", label='test (sim. id. result)')

        if not xy_id is None:
            x_pred = xy_id[:,0] - self.x0
            y_pred = xy_id[:,1] - self.y0
            line_res = self.ax_xy.plot(x_pred, y_pred, color=tudcolors.get("rood"), linestyle = "dotted", label='test (id. result)')


    def plot_histograms(self):

        if not self.plot_for_paper:
            rng = (-5,3)
            x = np.linspace(rng[0], rng[1], 100)

            for i in range(len(self.i_test)):
                if self.j_test[i] >= len(self.prediction_sample_metrics['LD'][0]):
                    continue
                ld = np.array(self.prediction_sample_metrics['LD'])[:,self.j_test[i]]

                hist_density, _, _ = self.axes_hist[i].hist(ld, range=rng, bins=70, density=True, color=tudcolors.get("donkergroen"))

                #density
                kde = gaussian_kde(ld)
                self.axes_hist[i].plot(x, kde(x), color=tudcolors.get("donkerblauw"))

                dmax = np.max(hist_density)
                ld_m = self.groundtruth_metrics['LD'][0][self.i_test[i]]
                self.axes_hist[i].plot((ld_m, ld_m),(0,dmax), color=tudcolors.get("rood"), zorder=100) 


    def beautify_plot(self):

        matplotlib.rcParams.update({"text.usetex": True})

        black = 'black'
        green = tudcolors.get('donkergroen')
        red = tudcolors.get('rood')
        
        fig = self.ax_xy.figure

        #delete histograms
        for ax in self.axes_hist:
            fig.delaxes(ax)

        #rearrange main plot
        gs = self.ax_xy.get_subplotspec().get_gridspec()
        self.ax_xy.set_position(gs[:,:].get_position(fig))
        self.ax_xy.set_subplotspec(gs[:,:])
            
        #adjust predictions
        linspec_pred = dict(linewidth=1, alpha=0.2)
        j0 = self.j_start_eval
        j1 = self.j_end_eval
        for l in self.plotted_lines['predictions']:
            
            # line
            line = l[0][0]
            lx = line.get_xdata()
            ly = line.get_ydata()
            line.remove()

            self.ax_xy.plot(lx[:j0+1], ly[:j0+1], color=black, **linspec_pred)
            self.ax_xy.plot(lx[j0:j1+1], ly[j0:j1+1], color=green, **linspec_pred)
            self.ax_xy.plot(lx[j1:], ly[j1:], color=black, **linspec_pred)
        
            # fde marker
            fde = l[-1][0]
            fde.set(markerfacecolor=green, markeredgecolor='none')

            # nll marker
            for nll in l[1:-1]:
                fde.set(color=green)

        if 'id-sim' in self.plotted_lines:
            self.plotted_lines['id-sim'][0].remove()

        if 'id-res' in self.plotted_lines:
            self.plotted_lines['id-res'][0].remove()

        pattern = r"(t_[a-z]{2})"
        for annotation in self.plotted_annotations:
            if re.findall(pattern, annotation):
                match = re.findall(pattern, annotation)[0]

                text = self.plotted_annotations[annotation][1].get_text()
                text = text.replace(match, r'$\tau_\mathrm{'+match[-2:]+r'}$')
                self.plotted_annotations[annotation][1].set_text(text)
                self.plotted_annotations[annotation][1].set_fontsize(10)
            elif self.plotted_annotations[annotation][1].get_text() != "":
                self.plotted_annotations[annotation][1].set_fontsize(10)
            else:
                self.plotted_annotations[annotation][0][0].remove()
                self.plotted_annotations[annotation][1].remove()

        fig.set_size_inches(2.23, 2)


    def savefig(self, dir_out, tag):
        """ Saves the xy figure of this prediction to dir_out/traj-predictions
        """

        dir_out = os.path.join(dir_out, f'{tag}_traj-predictions')
        if not os.path.isdir(dir_out):
            os.makedirs(dir_out)

        fig_xy_file_path = os.path.join(dir_out, f"pred-xy_{self.step_id}.png")
        fig = self.ax_xy.figure

        fig.savefig(fig_xy_file_path)





