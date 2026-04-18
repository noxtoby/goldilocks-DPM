###
# goldilocks_ZscoreSustain: 
#    a Python implementation of the goldilocks-dpm framework 
#    for z-score Subtype and Stage Inference (SuStaIn)
#
# If you use goldilocks-dpm, please cite the following paper:
# - The original Goldilocks DPM methods paper: TBA

# Please see https://github.com/ucl-pond/pySuStaIn for SuStaIn info and papers.
#
# Author:       Neil Oxtoby (gihub: noxtoby)
# Contributors: TBA
###
import warnings
from tqdm.auto import tqdm
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

from goldilocks_dpm import dpm_data,goldilocks_dpm

#*******************************************
#The data structure class for ZscoreSustain. It holds the z-scored data that gets passed around and re-indexed in places.
class goldilocks_ZscoreSustain_data(dpm_data):

    def __init__(self, data):
        self.data = data

    def get_num_samples(self):
        return self.data.shape[0]

    def get_num_biomarkers(self):
        return self.data.shape[0]

#*******************************************
# An implementation of the goldilocks_dpm class for the z-score event-based model, including SuStaIn.
class goldilocks_ZscoreSustain(goldilocks_dpm):

    def __init__(self,
                 classes,
                 sustain_object = None,
                 dpmData = None,
                 output_folder = None,
                 robust_zscores = True,
                 case_label = 1,
                 ctrl_label = 0,
                 direction_abnormal = None,
                 biomarker_labels = None
                 ):
        # Goldilocks-DPM initializer for a z-score event-based model (including SuStaIn)
        # Parameters:
        #   classes                     - categorical variable array indicating, e.g., diagnosis per sample. Used to convert data to z-scores.
        #                                 Assumes cases==1, controls==0 (non-cases and non-controls can be included as any other integer)
        #                                 dim: number of subjects x 1
        #   sustain_object              - a ZscoreSustain object
        #   dpmData                     - dim M x N: number of samples/subjects (rows) x number of biomarkers (columns)
        #   output_folder               - where to save outputs (plots)
        #   direction_abnormal          - direction of abnormality: automatically estimated from the data medians if None
        #   biomarker_labels            - the names of the biomarkers as a list of strings
        #
        # Outputs of run_goldilock_dpm():
        #   Z_vals                      - a matrix specifying recommended z-score event thresholds for each biomarker
        #                                 dim: M x N_events_max (maximum number of events across biomarkers)
        #                                 Follows pySuStaIn formatting where a ragged matrix is forward-filled with zeros
        #   Z_max                       - a vector specifying the maximum z-score for each biomarker (the "final event")
        #                                 dim: M x 1

        assert (classes is not None), f"Input variable classes cannot be None! Please provide a 1D array of numerical diagnostic classes for your data (with corresponding case_label and ctrl_label values)."

        # If supplied with a sustain_object, extract data, classes, output_folder
        if sustain_object:
            print("=== Creating goldilocks object from sustain object ===")
            dpmData = sustain_object.__dict__["_ZscoreSustain__sustainData"].data
            output_folder = sustain_object.__dict__["output_folder"]
        else:
            print("=== Creating goldilocks objects without sustain object ===")
            assert (dpmData is not None), f"Input variable dpmData cannot be None!"
            assert output_folder, f"Input variable output_folder cannot be None!"

        M, N                            = dpmData.shape # number of samples/subjects, biomarkers
        if biomarker_labels is None:
            biomarker_labels = [f'Biomarker {n}' for n in range(N)]
        assert (len(biomarker_labels) == N), "number of labels should match number of biomarkers"
        if direction_abnormal is None:
            # Automatically estimate direction of abnormality
            direction_abnormal = [1 if np.median(dpmData[classes==case_label,k])>np.median(dpmData[classes==ctrl_label,k]) else -1 for k in range(N)]
        assert case_label != ctrl_label, "case_label and ctrl_label cannot be identical"

        self.biomarker_labels   = biomarker_labels
        self.direction_abnormal = direction_abnormal
        self.robust_zscores     = robust_zscores
        self.case_label         = case_label
        self.ctrl_label         = ctrl_label

        self.__dpmData          = goldilocks_ZscoreSustain_data(dpmData)

        super().__init__(dpmData = self.__dpmData,
                         classes = classes,
                         output_folder = output_folder,
                         robust_zscores = robust_zscores,
                         case_label = case_label,
                         ctrl_label = ctrl_label,
                         direction_abnormal = direction_abnormal,
                         biomarker_labels = biomarker_labels)

    #= public methods
    def run_goldilocks(self, plot=True, plot_format="png", verbose=True, **kwargs):
        z_vals_default = [0.67,1.65,3.1] # 75%, 95%, 99.9% CDF
        Z_vals                 = []
        Z_max                  = []
        ## Normalise data
        if self.robust_zscores:
            Z, avg, spread = goldilocks_dpm.zscore_robust(self.__dpmData.data, self.classes, self.ctrl_label, self.direction_abnormal)
        else:
            Z, avg, spread = goldilocks_dpm.zscore(self.__dpmData.data, self.classes, self.ctrl_label, self.direction_abnormal)
        self.Z = Z
        # print(f"avg.shape: {avg.shape}")
        # print(f"spread.shape: {spread.shape}")
        # print(f"Z.shape: {Z.shape}")
        ## Calculate Goldilocks Zone
        # Empirical distribution
        y = self.classes
        Z_not_controls = Z[y!=self.ctrl_label,:]
        y_not_controls = y[y!=self.ctrl_label,]
        # print(f"sum(y_not_controls==1): {sum(y_not_controls==1)}")
        # print(y_not_controls)
        edf = np.empty(Z_not_controls.shape)
        ecdf = []
        if plot:
            fig,ax = plt.subplots(1,len(self.biomarker_labels),figsize=(12,3),sharey=True,sharex=True)
        for k in range(len(self.biomarker_labels)):
            ecdf += [sm.distributions.empirical_distribution.ECDF(Z_not_controls[:,k])]
            edf[:,k] = ecdf[k](Z_not_controls[:,k])
            if plot:
                ax[k].plot(Z_not_controls[:,k],edf[:,k],'.')
                ax[k].set_title(self.biomarker_labels[k],fontsize=17)
                ax[k].set_xlabel('Z',fontsize=17)
        if plot:
            ax[0].set_ylabel('EDF(Z)',fontsize=17)
            fig.show()
        ## Event Horizon 1: minimum event threshold.
        # where the z score in non-controls exceeds at least half of the positive (abnormal) z-scores in controls (ignore the negative/more normal values),
        # i.e., the minimum event threshold can be defined in terms of the half-normal CDF:
        #    arg-min |CDF_{half-normal}(z_min)-0.5|
        # which is where the positive z-scores from non-controls are "more abnormal" than the corresponding data from controls
        lambda_z_min = lambda x,c: np.argmin(np.abs(stats.halfnorm.cdf(x)-c))
        z_ = np.arange(-5,5,0.001) # +/- 3 sigma should cover most data from controls
        t = 0.5
        z_min = [ z_[lambda_z_min(z_,t)] ]*Z.shape[1]
        if verbose:
            print("Goldilocks Event Horizons: you might consider omitting features where the median in cases does not exceed z_min.")
            print(f"Your data:\n medians: {np.around(np.nanmedian(Z_not_controls[y_not_controls==1,:],axis=0),2)}\n z_min:   {np.around(z_min,2)}")
            print(" === Event Horizon 2: z_max === ")
        ## Event Horizon 2: maximum event threshold.
        # Some proportion of data must exceed the maximum z-score event threshold, 
        # or the model cannot be informative at this level.
        # Here this proportion is set at 10%, i.e., the EDF must exceed 0.9, or:
        #    arg-min |EDF_{half-normal}(z_min)-0.9|
        edf_cutoff = 0.9
        lambda_z_max = lambda e,t: np.argmin(np.abs(e-t))
        z_max = [Z_not_controls[lambda_z_max(edf[:,k],edf_cutoff),k] for k in range(edf.shape[1])]
        if verbose:
            print("Goldilocks Event Horizons: you should consider the maximum value your data can support, which should inform hyperparameters of your model.")
            print(f" Here we've used an EDF cutoff of {edf_cutoff} to define `z_max`")
            print(f"Your data:\n max: {np.around(np.nanmax(Z_not_controls[y_not_controls==1,:],axis=0),2)}\n z_max:   {np.around(z_max,2)}")
        event_horizons_ = [z_min,z_max]
        # Plot
        if plot:
            fig,ax = plt.subplots(1,Z.shape[1],figsize=(12,3),sharey=True)
            for k in range(len(self.biomarker_labels)):
                ax[k].plot(Z_not_controls[:,k],edf[:,k],'.')
                ax[k].set_title(self.biomarker_labels[k],fontsize=17)
                ax[k].set_xlabel('Z',fontsize=17)
                ax[k].plot([z_min[k],z_min[k]],[0,1],'g-')
                ax[k].plot([z_max[k],z_max[k]],[0,1],'r:')
                zed = np.linspace(z_min[k],z_max[k],100) #zed = np.arange(z_min[k],z_max[k],0.05)
                ax[k].fill_between(zed,ecdf[k](zed),color='gold')
                # ax[k].plot(Z_not_controls[:,k],np.abs(edf[:,k]-edf_cutoff),'rx')
            ax[0].set_ylabel('EDF(Z)',fontsize=17)
            fig.tight_layout()
            fig.show()
        ## Middle event thresholds for `ZscoreSustain`
        # Here we take a simple average, but you might want to do something more 
        # clever, e.g.,:
        # - map Gaussian to uniform/linear via Inverse Quantile transformation
        # - something in log space like the logarithmic mean or the geometric mean
        # - fixed thresholds corresponding to convenient CDF values, e.g., 
        #   - z ~ 0.67 == CDF=.75  (abnormality beyond 75%   of controls)
        #   - z ~ 1.65 == CDF=.95  (abnormality beyond 95%   of controls)
        #   - z ~ 3.1  == CDF=.999 (abnormality beyond 99.9% of controls)
        midpoints = np.mean(event_horizons_,axis=0)
        rng = np.random.RandomState(0)
        qt = QuantileTransformer(n_quantiles=10, random_state=0)
        X = np.sort(rng.normal(loc=0, scale=1.0, size=(200, 1)), axis=0)
        y = qt.fit_transform(X)
        Y = norm.ppf(y)
        Y_trans = (Y - Y[1])
        Y_trans = Y_trans/Y_trans[-2]
        #y = (norm.cdf(x)-0.5)/0.5
        #y_trans = (norm.cdf(np.log1p(x))-0.5)/0.5
        ##Y = (qt.fit_transform(X)-0.5)/0.5
        #Y = qt.fit_transform(y.reshape(-1,1))
        b = X>=0
        x = X[b]
        y = y[b]
        y_trans = Y_trans[b]
        fig,ax=plt.subplots()
        #ax.plot(X[X>=0],Y[X>=0],label='inverse quantile')
        ax.plot(x,y,label='Quantile transform')
        ax.plot(x,y_trans,label='PPF of Quantile (normed)')
        #ax.plot(X[X>=0],Y[X>=0],label='inverse quantile')
        ax.legend()
        fig.show()
        ## END middle event thresholds
        if verbose:
            print("If you want to add events between z_min and z_max, then you might take an average (for example).")
            print(f" Here we get:\n  midpoints: {np.around(midpoints,2)}")
        # Round the event horizons
        fraction_to_round = 0.25
        event_horizons = np.concatenate(([[np.around(z/fraction_to_round)*fraction_to_round for z in z_min]],
                                         [[np.around(z/fraction_to_round)*fraction_to_round for z in z_max]]),axis=0)
        if verbose:
            print(f"event_horizons rounded: \n{event_horizons}")
        # Round the midpoints
        midpoints = np.around(midpoints/fraction_to_round)*fraction_to_round
        if verbose:
            print(f"midpoints rounded: \n{midpoints}")
        ## ZScoreSustain recommendations
        if verbose:
            print("The event horizons and midpoint event thresholds define sensible Z_vals hyperparameters for ZscoreSustain:")
        # Z_vals, including midpoint event thresholds
        Z_vals = np.concatenate(
            ( [event_horizons[0]],
              [midpoints],
              [event_horizons[1]] ),
            axis=0
        ).T
        if verbose:
            print(f"Z_vals = \n{Z_vals}")
            print("\n...and the rounded maximum values for each feature can inform Z_max hyperparameters for ZscoreSustain:")
        # Z_max - simply round up
        Z_max = [np.around(np.nanmax(Z_not_controls[:,k])/fraction_to_round)*fraction_to_round for k in range(Z_not_controls.shape[1])]
        #Z_max = np.array([Z_max]).T
        if verbose:
            print(f"Z_max = \n{Z_max}")
        if verbose:
            self._zvals_note()

        ## Translate the Goldilocks Zone z-values back to raw data
        lambda_z_to_x = lambda z,m,s,d: (z*s)/d + m
        abnorm_dir  = np.reshape(np.array(self.direction_abnormal),(-1,1))
        avg_vals    = np.reshape(avg[0,],(-1,1))
        spread_vals = np.reshape(spread[0,],(-1,1))
        abnorm_dir  = np.tile(abnorm_dir  ,(1,Z_vals.shape[1]))
        avg_vals    = np.tile(avg_vals    ,(1,Z_vals.shape[1]))
        spread_vals = np.tile(spread_vals ,(1,Z_vals.shape[1]))
        X_vals = lambda_z_to_x(Z_vals,avg_vals,spread_vals,abnorm_dir)
        mu   = avg_vals[:,0]    #np.reshape(avg_vals[:,0],(-1,1))
        sig  = spread_vals[:,0] #np.reshape(spread_vals[:,0],(-1,1))
        dirn = abnorm_dir[:,0]  #np.reshape(abnorm_dir[:,0],(-1,1))
        X_max  = lambda_z_to_x(Z_max, mu, sig, dirn)
        goldilocks_zone = X_vals[:,[0,-1]]
        self.Z_vals = Z_vals
        self.Z_max  = Z_max
        self.X_vals = X_vals
        self.X_max  = X_max
        print("Goldilocks Zone calculated. See recommended Z_vals and Z_max within your goldilocks_ZScoreSustain object.")

        return goldilocks_zone

    def _zvals_note(self):
        # A note for the user
        print("""
        # NOTE: Visual interpretation of subtype progression patterns ("positional variance diagrams") 
                estimated with `ZscoreSustain` are potentially simpler when `Z_vals` are the same for 
                each biomarker.
                
                This is because the plotting method in pySuStaIn was originally hard-coded for the case 
                where Z_vals = [1,2,3] for each feature/biomarker: 1=red, 2=magenta, 3=blue.
                
                The function _plot_goldilocks_ZScoreSustain_pvd() is provided here to help with this, 
                but it may not completely suit your needs.
                
                Alternatively, to ease model visualisation and interpretation, you might want to 
                manually intervene after running Goldilocks by selecting a compromise, e.g.,
                
                ```
                Z_vals = 
                [[0.75 1.25 1.75]                       [[0.75  1.5 2. ]
                 [0.75 3.   5.  ]                        [0.75  1.5 2. ]
                 [1.   3.75 6.75]       could become     [0.75  1.5 2. ]
                 [0.75 3.5  6.  ]                        [0.75  1.5 2. ]
                 [0.75 2.5  4.  ]]                       [0.75  1.5 2. ]]
                ```
                
                and/or you could contribute to improving the plotting module in `pySuStaIn`. :-)
        
        ~ Neil Oxtoby, UCL, November 2023
        """)

    def _plot_goldilocks_ZScoreSustain_pvd(samples_sequence,
                                           samples_f,
                                           biomarker_labels,
                                           stage_zscore,
                                           stage_biomarker_index,
                                           colour_mat,
                                           plot_order,
                                           subtype_labels,
                                           all_zscores):
        '''
        _plot_goldilocks_ZScoreSustain_pvd()
        
        Provides basic plotting functionality for a ZScoreSustain model where Z_vals vary per biomarker.
        
        See also _zvals_note()
        
        Acknowledgement:
        - Adapted by Neil Oxtoby in Nov 2023 from MATLAB code generously shared by Alex Young.
        '''
        temp_mean_f = np.mean(samples_f, axis=1)
        vals_ix = np.argsort(temp_mean_f)[::-1]
        vals = np.round(temp_mean_f[vals_ix], 2)

        N_S = samples_sequence.shape[0]

        if N_S <= 3:
            fig = plt.figure(figsize=(50 * N_S, 40))
        else:
            fig = plt.figure(figsize=(50 * np.ceil(N_S / 2), 45 * np.floor(N_S / 2)))

        for i in range(N_S):
            this_samples_sequence = samples_sequence[vals_ix[i],:,:].T

            if N_S <= 3:
                ax = plt.subplot(1, N_S, i + 1)
            else:
                ax = plt.subplot(np.floor(N_S / 2), np.ceil(N_S / 2), i + 1)

            zvalues = np.unique(stage_zscore)
            colour_mat_rows = np.array([True if z in zvalues else False for z in all_zscores])
            this_colour_mat = colour_mat[colour_mat_rows,:]
            markers = np.unique(stage_biomarker_index)
            N_z = this_colour_mat.shape[0]
            N = this_samples_sequence.shape[1]

            confus_matrix = np.zeros((N, N))
            for j in range(N):
                confus_matrix[j, :] = np.sum(this_samples_sequence == j, axis=0)

            confus_matrix /= np.max(this_samples_sequence.shape)

            N_bio = markers.shape[0]
            confus_matrix_z = np.zeros((N_bio, N, N_z))
            for z in range(N_z):
                true_false = stage_zscore == zvalues[z]
                confus_matrix_z[stage_biomarker_index[stage_zscore == zvalues[z]], :, z] = confus_matrix[true_false.flatten(), :]
            confus_matrix_c = np.ones((N_bio, N, 3))
            for z in range(N_z):
                this_colour = this_colour_mat[z, 0:-1]
                this_confus_matrix = confus_matrix_z[:, :, z]
                this_colour_matrix = np.repeat(this_confus_matrix[markers, :][:, :, np.newaxis], 3, axis=2) * (1 - this_colour)[np.newaxis, np.newaxis, :]
                confus_matrix_c -= this_colour_matrix

            im = ax.imshow(confus_matrix_c[plot_order, :, :])
            ax.set_xticks(np.arange(5, N, 5), labels=np.arange(5, N, 5))
            ax.set_yticks(np.arange(N_bio), labels=[biomarker_labels[k] for k in plot_order])
            ax.set_xlabel('SuStaIn Stage')
            ax.set_title(subtype_labels[i])
        cbar=plt.colorbar(im)

        fig.show()
        return fig,ax,cbar

    def _plot_goldilocks(self, *args, **kwargs):
        return goldilocks_ZScoreSustain.plot_goldilocks_zone(*args, Z_vals=self.Z_vals, **kwargs)

    # # ********************* STATIC METHODS
    # @staticmethod
    # def zscore_robust(X, y, ctrl_label):
    #     c = y==ctrl_label
    #     return (X - np.nanmedian(X[c,:],axis=0))/stats.median_abs_deviation(X[c,:],axis=0)
    #
    # @staticmethod
    # def zscore(X, y, ctrl_label):
    #     c = y==ctrl_label
    #     return (X - np.nanmean(X[c,:],axis=0))/np.std(X[c,:],axis=0)
