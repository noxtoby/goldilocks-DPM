#!/usr/bin/env python3
#
# Goldilocks-DPM framework
#
# Neil Oxtoby, UCL, 2023
#


from abc import ABC, abstractmethod

# from tqdm.auto import tqdm
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
import statsmodels.api as sm
# import matplotlib.colors as mcolors
# from pathlib import Path
# import pickle
# import csv
# import os
# import multiprocessing
# from functools import partial, partialmethod
#
# import time
# import pathos


#= Abstractions: define your own implementations

#************
class dpm_data(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_num_samples(self):
        pass

    @abstractmethod
    def get_num_biomarkers(self):
        pass

#************
class goldilocks_dpm(ABC):

    def __init__(self,
                 dpmData,
                 classes,
                 output_folder,
                 robust_zscores = True,
                 case_label = 1,
                 ctrl_label = 0,
                 direction_abnormal = None,
                 biomarker_labels = None
                 ):
        # The initializer for the abstract class
        # Parameters:
        #   dpmData           - an instance of the dpm_data class

        assert(isinstance(dpmData, dpm_data))

        self.__dpmData              = dpmData
        self.classes                = classes
        self.output_folder          = output_folder
        self.robust_zscores         = robust_zscores
        self.case_label             = case_label
        self.ctrl_label             = ctrl_label
        self.direction_abnormal     = direction_abnormal
        self.biomarker_labels       = biomarker_labels

    def run_goldilocks(self, plot=True, plot_format="png", verbose=True, **kwargs):
        pass


    # ********************* STATIC METHODS
    @staticmethod
    def zscore_robust(X,y,ctrl_label,abnormal_direction):
        c = y==ctrl_label
        avg = np.nanmedian(X[c,:],axis=0)
        spread = stats.median_abs_deviation(X[c,:],axis=0)
        avg = np.tile(avg, (X.shape[0],1))
        spread = np.tile(spread, (X.shape[0],1))
        Z = abnormal_direction*(X - avg)/spread
        return Z, avg, spread

    @staticmethod
    def zscore(X,y,ctrl_label,abnormal_direction):
        c = y==ctrl_label
        avg = np.nanmean(X[c,:],axis=0)
        spread = np.std(X[c,:],axis=0)
        avg = np.tile(avg, (X.shape[0],1))
        spread = np.tile(spread, (X.shape[0],1))
        Z = abnormal_direction*(X - avg)/spread
        return Z, avg, spread
        

# Synthesize z-score data
# - Sample uniformly across multivariate space => should produce uniform staging in SuStaIn subtypes

# Start with default waypoints z=1,2,3

# Calculate biomarker Goldilocks Zones and new waypoints
# => z_min, z_max from Goldilocks event horizons
# => z_mid recommendations:
#    1. Geometric Midpoint
#    2. Vogel-style mixture modelling (as optional input parameter for the goldilocks() function)

# Move the waypoints and output as Z matrix for pySuStaIn
