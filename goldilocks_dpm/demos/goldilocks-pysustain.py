#!/usr/bin/env python3
# Demonstration of Goldilocks DPM framework for pySuStaIn
# Neil Oxtoby, UCL, 2023

run_locally = False
rundate = '20231211.2'

if rundate=='20231211':
    dataset_name  = "SynthADNI"
    config        = "vanilla"
    nom           = ""
elif rundate=='20231211.2':
    dataset_name  = "SynthADNI"
    config        = "goldilocks"
    nom           = "_rerun"

from pathlib import Path
path_to_goldilocks_dpm = Path.cwd().parent / "goldilocks-dpm"
path_to_demos = path_to_goldilocks_dpm / "goldilocks_dpm" / "demos"
output_folder = path_to_demos / "cluster"

import sys,os
# sys.path.append(path_to_goldilocks_dpm)
os.chdir(path_to_goldilocks_dpm)

import pandas as pd, numpy as np, seaborn as sn
from matplotlib import pyplot as plt

from goldilocks_dpm import goldilocks_ZscoreSustain

from pySuStaIn.ZscoreSustain  import ZscoreSustain

# Load synthetic data
csv_main = path_to_demos / "ADNIMERGE2023_synthetic.csv"
assert csv_main.exists(), f"ERROR: {csv_main.name} file not found."
df = pd.read_csv(csv_main,low_memory=False)

# Save CV indices for cluster
import pickle
from sklearn.model_selection import StratifiedKFold
N_folds = 10
pickle_filename_cv = output_folder / csv_main.name.split('/')[-1].replace('.csv','-cvindices.pickle')

biomarkers         = ["ABETA","TAU","ADAS13","MMSE","Ventricles_ICV"]
direction_abnormal = [     -1,    1,       1,    -1,               1]

# Patients and controls
dx_col = "DX"
id_col = "ID"
df['y'] = df[dx_col].values
controls  = (df[dx_col]==0).values
cases     = (df[dx_col]==1).values
prodromal = (df[dx_col]==2).values

# Raw data
X_synth = df[biomarkers].values
y_synth = df[dx_col].values
id_synth = df[id_col].values
print(f"Your data contains {X_synth.shape[0]} samples from {X_synth.shape[1]} biomarkers: \n{biomarkers}")

# Create Goldilocks DPM object, version 1: without a pySuStaIn object
gdpm = goldilocks_ZscoreSustain(
    classes = y_synth,
    dpmData = X_synth,
    output_folder = output_folder,
    robust_zscores = False,
    case_label = 1,
    ctrl_label = 0, 
    direction_abnormal = direction_abnormal,
    biomarker_labels = biomarkers
)

# Scroll down to "ALTERNATIVE" for how to create Goldilocks DPM object using an existing pySuStaIn object

gdpm.run_goldilocks(plot=True, plot_format="png", verbose=False)

print(gdpm.Z_vals)
print(gdpm.Z_max)

# Add z-scores to dataframe
sustain_features = [f"{b}_z" for b in biomarkers]
Z_synth = gdpm.Z.copy()
df[sustain_features] = Z_synth

data = Z_synth
y    = y_synth
id_  = id_synth

sustain_output_folder = '_'.join( [dataset_name,'zscore',rundate] )
sequence_without_controls = True
sustainType             = 'zscore'
y_for_sequencing = list(df[dx_col].values!=0)
if sequence_without_controls:
    df_train = df.loc[y_for_sequencing].copy()
    data_sequencing = data[y_for_sequencing,:]
    y_sequencing    = y[y_for_sequencing,]
    id_sequencing   = id_[(y_for_sequencing)]
    sustain_output_folder = output_folder / f'{sustain_output_folder}_sequence_without_controls'
else:
    df_train = df.copy()
    data_sequencing = data
    y_sequencing    = y
    id_sequencing   = id_
    sustain_output_folder = output_folder / sustain_output_folder
os.system(f'mkdir -p {sustain_output_folder}')


# number of starting points
N_startpoints           = 25
# maximum number of subtypes
N_S_max                 = 4
N_iterations_MCMC       = int(1e4)  #int(1e6)
# cross-validation
validate                = True
N_folds                 = 10

M_seq, N_seq            = data_sequencing.shape     # number of individuals, events

## pySuStaIn v1: defaults
z_events                = [1,2,3]
Z_vals_default          = np.array([z_events]*N_seq) # Z-scores for each biomarker
Z_max_default           = np.array([5]*N_seq)        # maximum z-score

## Goldilocks-informed pySuStaIn
Z_vals_goldilocks       = gdpm.Z_vals                # Z-scores for each biomarker
Z_max_goldilocks        = np.array(gdpm.Z_max)       # maximum z-score

# Avoid Z_max < Z_vals
Z_max_default    = np.max([np.reshape(np.max(Z_vals_default,axis=1),(-1,1)),np.reshape(Z_max_default,(5,1))],axis=0).flatten()
Z_max_goldilocks = np.max([np.reshape(np.max(Z_vals_goldilocks,axis=1),(-1,1)),np.reshape(Z_max_goldilocks,(5,1))],axis=0).flatten()


sustain_default = ZscoreSustain(
    data_sequencing, 
    Z_vals_default, Z_max_default, 
    biomarkers, 
    N_startpoints, N_S_max, N_iterations_MCMC, 
    output_folder, 'SynthADNI_sustain_default',
    use_parallel_startpoints=False, seed=42
)

sustain_goldilocks = ZscoreSustain(
    data_sequencing,
    Z_vals_goldilocks, Z_max_goldilocks,
    biomarkers,
    N_startpoints, N_S_max, N_iterations_MCMC,
    output_folder, 'SynthADNI_sustain_goldilocks',
    use_parallel_startpoints=False, seed=42
)

## ALTERNATIVE: create Goldilocks DPM object directly from a pySuStaIn object
## TODO: make this work.
# gdpm_alt = goldilocks_ZscoreSustain(
#     sustain_object = sustain_default,
#     #dpmData = None,
#     classes = y_sequencing,
#     output_folder = output_folder,
#     robust_zscores = False,
#     case_label = 1,
#     ctrl_label = 0,
#     direction_abnormal = direction_abnormal,
#     biomarker_labels = biomarkers
# )
# gdpm_alt.run_goldilocks(plot=True, plot_format="png", verbose=False)



if ~run_locally:
    print("Preparing data for SuStaIn to be run elsewhere, e.g., on a compute cluster")
    # Training data
    if sum(df_train['y'].isnull())==0:
        df_train['y'] = df_train['y'].astype(int).values
    else:
        print(f'Missing data in y (cases/controls labels): df_train["y"].isnull().sum(): {df_train["y"].isnull().sum()} (of {df_train["y"].shape[0]}):')
        print(df_train.loc[df_train["y"].isnull()].groupby('Diagnosis')[id_col].count())
    # Write out to CSV for the cluster
    colz = ['y',id_col,dx_col] + sustain_features
    colz = colz + [c for c in df_train.columns.tolist() if c not in colz]
    df[colz].to_csv(str(csv_main).replace('.csv','_wrangled-all.csv'),index=False)
    df_train[colz].to_csv(str(csv_main).replace('.csv','_wrangled-train.csv'),index=False)
else:
    # Don't worry, if run_locally==True, SuStaIn is run further down if it hasn't been run already
    print("")



if pickle_filename_cv.exists():
    pickle_file                 = open(pickle_filename_cv, 'rb')
    variables_to_pickle         = pickle.load(pickle_file)
    test_idxs = variables_to_pickle['test_idxs']
    pickle_file.close()
    for k in range(len(test_idxs)):
        id_train = id_sequencing[test_idxs[k]] #[id_sequencing[j] for j in t]
        rowz = df["ID"].isin(id_train).values
        df.loc[rowz,'cv_idx'] = k
else:
    df['cv_idx'] = np.nan
    #* k-fold cross validation
    test_idxs  = []
    train_idxs  = []
    cv         = StratifiedKFold(n_splits=N_folds, shuffle=True)
    cv_it      = cv.split(data_sequencing, y_sequencing)
    
    k = 0
    for train, test in cv_it:
        test_idxs.append(test)
        train_idxs.append(train)
        k += 1
    test_idxs = np.array(test_idxs,dtype='object')
    train_idxs = np.array(train_idxs,dtype='object')
    
    for k in range(len(test_idxs)):
        id_train = id_sequencing[test_idxs[k]] #[id_sequencing[j] for j in t]
        rowz = df["ID"].isin(id_train).values
        df.loc[rowz,'cv_idx'] = k
    
    pickle_filepath             = Path(pickle_filename_cv)
    pickle_file                 = open(pickle_filename_cv, 'wb')
    variables_to_pickle         = {
        'test_idxs': test_idxs
    }
    po = pickle.dump(variables_to_pickle,pickle_file)
    pickle_file.close()





if run_locally:
    ## Default SuStaIn
    # get the start time
    st_default = time.process_time()
    sustain_default.run_sustain_algorithm(plot=False)
    # get the end time
    et_default = time.process_time()
    # get execution time
    res_default = et_default - st_default
    print('Default SuStaIn CPU Execution time:', res_default/60, 'minutes')


    ## Goldilocks SuStaIn
    # get the start time
    st_goldilocks = time.process_time()
    sustain_goldilocks.run_sustain_algorithm(plot=False) # plot=True gives an error: pySuStaIn plotting only handles identical z-event sets per biomarker
    # get the end time
    et_goldilocks = time.process_time()
    # get execution time
    res_goldilocks = et_goldilocks - st_goldilocks
    print('Goldilocks-informed SuStaIn CPU Execution time:', res_goldilocks/60, 'minutes')





if run_locally:
    ## TODO: plot staging distributions => should get better (flatter) spread for Goldilocks
    data_inference = gdpm.Z
    y_inference = gdpm.classes
    s = 1

    pickle_filename_s           = output_folder + '/pickle_files/' + sustain_default.dataset_name + '_subtype' + str(s) + '.pickle'
    pickle_filepath             = Path(pickle_filename_s)
    pickle_file                 = open(pickle_filename_s, 'rb')
    loaded_variables            = pickle.load(pickle_file)
    ml_subtype                  = loaded_variables["ml_subtype"]
    prob_ml_subtype             = loaded_variables["prob_ml_subtype"]
    ml_stage                    = loaded_variables["ml_stage"]
    prob_ml_stage               = loaded_variables["prob_ml_stage"]
    prob_subtype                = loaded_variables["prob_subtype"]
    prob_stage                  = loaded_variables["prob_stage"]
    prob_subtype_stage          = loaded_variables["prob_subtype_stage"]
    samples_sequence            = loaded_variables["samples_sequence"]
    samples_f                   = loaded_variables["samples_f"]
    pickle_file.close()

    N_samples            = data_inference.shape[0]
    ml_subtype,          \
    prob_ml_subtype,     \
    ml_stage,            \
    prob_ml_stage,       \
    prob_subtype,        \
    prob_stage,          \
    prob_subtype_stage   = sustain_default.subtype_and_stage_individuals_newData(
        gdpm.Z,
        samples_sequence,
        samples_f,
        N_samples
    )

    # Test out new plotting code from ChatGPT
    all_zscores = np.unique(np.sort(list(sustain_goldilocks.stage_zscore) + list(sustain_default.stage_zscore)))
    # import matplotlib.colors as mcolors
    # print(list(mcolors.XKCD_COLORS.keys())[0:len(all_zscores)])

    from matplotlib import cm
    viridis = cm.get_cmap('viridis', len(all_zscores))
    colour_mat_all = viridis(all_zscores/all_zscores.max())
    # Select subset of colours
    zscores_defaults   = np.unique(sustain_default.stage_zscore)
    zscores_goldilocks = np.unique(sustain_goldilocks.stage_zscore)
    rowz_default    = [True if z in zscores_defaults   else False for z in all_zscores]
    rowz_goldilocks = [True if z in zscores_goldilocks else False for z in all_zscores]
    colour_mat_default    = colour_mat_all[rowz_default,   :]
    colour_mat_goldilocks = colour_mat_all[rowz_goldilocks,:]

    plot_order = np.arange(len(biomarkers))

    subtype_labels = ['0','1']

    fig,ax,cbar = plot_SuStaIn_model_arbitrarycolours(
        samples_sequence, 
        samples_f,
        biomarkers,
        sustain_goldilocks.stage_zscore,
        sustain_goldilocks.stage_biomarker_index,
        colour_mat_all,
        plot_order,
        subtype_labels,
        all_zscores
    )
    xt = all_zscores/np.max(all_zscores)
    xtl = [str(k) for k in all_zscores]
    cbar.set_ticks(xt)
    cbar.set_ticklabels(xtl)


    #* Plot subtypes and stages by diagnostic group
    stages_bins = np.arange(-0.5,1+data_inference.shape[1]*np.max(Z_max_default))
    # dx_list = np.sort(list(dx_column_mapper.keys()))
    # dx_list_label = [dx_column_mapper[dx][-1] for dx in dx_list]
    dx_list = np.unique(df[dx_col].values)
    dx_list_label = [str(dx) for dx in dx_list]
    fig,ax = plt.subplots(2,s+1,figsize=(16,8),sharey=False)
    axflat = ax.flatten()
    for k in range(s):
        st_k = (ml_subtype==k).flatten()
        st_nonzero = (ml_stage>0).flatten()
        ax[0,k].hist(y_inference[st_k & st_nonzero],bins=np.arange(-0.5,max(dx_list)+0.5))
        ax[0,k].set_xticks(dx_list)
        ax[0,k].set_xticklabels(dx_list_label,fontsize=18)
        ax[0,k].set_title('Subtype %i' % (k+1),fontsize=20)
        stages_tmp = []
        for dx_k in dx_list:
            if sum((y_inference==dx_k))==0:
                continue
            else:
                # FIXME: handle empty cases where no individuals with dx_k are subtyped/staged
                ax[1,k].hist( ml_stage[st_k & st_nonzero & (y_inference==dx_k)], label=dx_list_label,bins=stages_bins)
                ax[1,k].set_xlabel('Stage',fontsize=20)
    ax.flatten()[0].set_ylabel('Count',fontsize=20)
    ax.flatten()[0].set_xlabel('DX',fontsize=20)
    # ax[1,0].set_ylabel('Count',fontsize=20)
    # ax[1,0].set_ylim([0,50])
    ax.flatten()[0].legend(fontsize=20)
    fig.tight_layout()
    fig.show()
