import numpy as np
import matplotlib.pyplot as plt

def plot_SuStaIn_model_arbitrarycolours(samples_sequence, samples_f,
                                       biomarker_labels, stage_zscore,
                                       stage_biomarker_index, colour_mat,
                                       plot_order, subtype_labels, all_zscores):
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
            # Equivalent to subplots(sharey=True)
            if i>0:
                ax = plt.subplot(1, N_S+1, i + 1,sharey=ax)
            else:
                ax = plt.subplot(1, N_S+1, i + 1)
        else:
            if i>0:
                ax = plt.subplot(np.floor(N_S / 2), np.ceil(N_S / 2), i + 1,sharey=ax)
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
            #this_colour_matrix = np.repeat(this_confus_matrix[markers, :][:, :, np.newaxis], N, axis=2) * (1 - this_colour)[np.newaxis, np.newaxis, :]
            this_colour_matrix = np.repeat(this_confus_matrix[markers, :][:, :, np.newaxis], 3, axis=2) * (1 - this_colour)[np.newaxis, np.newaxis, :]
            confus_matrix_c -= this_colour_matrix

        im = ax.imshow(confus_matrix_c[plot_order, :, :])
        ax.set_xticks(np.arange(5, N, 5), labels=np.arange(5, N, 5))
        ax.set_yticks(np.arange(N_bio), labels=[biomarker_labels[k] for k in plot_order])
        ax.set_xlabel('SuStaIn Stage')
        ax.set_title(subtype_labels[i])
    # Add colour bar to next subplot
    if N_S <= 3:
        ax = plt.subplot(1, N_S+1, i + 2, sharey=ax)
    else:
        ax = plt.subplot(np.floor(N_S / 2), np.ceil(N_S / 2), i + 2, sharey=ax)
    cbar = plt.colorbar(im)

    fig.show()
    return fig,ax,cbar

# Example usage:
# Replace the following placeholders with your actual data
# plot_SuStaIn_model_arbitrarycolours(samples_sequence, samples_f,
#                                     biomarker_labels, stage_zscore,
#                                     stage_biomarker_index, colour_mat,
#                                     plot_order, subtype_labels)
