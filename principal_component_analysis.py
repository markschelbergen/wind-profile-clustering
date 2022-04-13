from sklearn.decomposition import PCA
from itertools import accumulate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from wind_resource_fit import log_law_wind_profile2

xlim_pc12 = [-1.1, 1.1]
ylim_pc12 = [-1.1, 1.1]
x_lim_profiles = [-0.8, 1.25]


def plot_mean_and_pc_profiles(altitudes, var, get_profile):
    plot_n_pcs = 2
    n_cols = 3
    n_rows = plot_n_pcs
    shape_map = (n_rows, n_cols)

    x_label = r"$\tilde{v}$ [-]"
    figsize = (8.4, 6)

    fig, ax = plt.subplots(shape_map[0], shape_map[1], sharey=True, figsize=figsize)
    wspace = 0.1
    layout = {'top': 0.9, 'bottom': 0.085, 'left': 0.38, 'right': 0.985, 'hspace': 0.23}
    plt.subplots_adjust(**layout, wspace=wspace)

    # Add plot window for mean wind profile to existing plot windows and plot it.
    w, h, y0, x0 = ax[0, 0]._position.width, ax[0, 0]._position.height, ax[0, 0]._position.y0, ax[0, 0]._position.x0
    ax_mean = fig.add_axes([x0-w*(1.5), y0, w, h])

    v = get_profile()
    ax_mean.plot(v, altitudes, label="Parallel", color='#ff7f0e')
    # ax_mean.plot(prp, altitudes, label="Perpendicular", color='#1f77b4')
    # ax_mean.plot((prl**2 + prp**2)**.5, altitudes, '--', label='Magnitude', ms=3, color='#2ca02c')
    ax_mean.set_title("Mean")
    ax_mean.grid(True)
    ax_mean.set_ylabel("Height [m]")
    ax_mean.set_xlim(x_lim_profiles)
    ax_mean.set_xlabel(x_label)
    ax_mean.legend(bbox_to_anchor=(1., 1.16, 3, 0.2), loc="lower left", mode="expand",
                borderaxespad=0, ncol=4)

    # # Add plot window for hodograph to existing plot windows and plot it.
    # y0 = ax[1, 0]._position.y0 + .05
    # ax_tv = fig.add_axes([x0-w*(1.5), y0, w, h])
    # ax_tv.plot(prl, prp, color='#7f7f7f')
    # ax_tv.plot([0, prl[0]], [0, prp[0]], 'b:', color='#7f7f7f')
    # ax_tv.grid(True)
    # ax_tv.axes.set_aspect('equal')
    # ax_tv.set_xlabel(r"$\tilde{v}_{\parallel}$ [-]")
    # ax_tv.set_ylabel(r"$\tilde{v}_{\bot}$ [-]")
    # ax_tv.set_xlim(x_lim_profiles)
    # ax_tv.set_ylim([-.3, .3])

    # Plot PCs and PC multiplicands superimposed on the mean.
    marker_counter = 0
    for i_pc in range(plot_n_pcs):  # For every PC/row in the plot.
        ax[i_pc, 0].set_ylabel("Height [m]")
        std = var[i_pc]**.5
        factors = iter([-1*std, 1*std])

        for i_col in range(n_cols):
            # Get profile data.
            if i_col == 0:  # Column showing solely the PCs
                v = get_profile(i_pc, 1, True)
                ax[i_pc, i_col].set_xlim(xlim_pc12)
                ax[i_pc, i_col].set_title("PC{}".format(i_pc+1))
            else:  # Columns showing PC multiplicands superimposed on the mean.
                factor = next(factors)
                v = get_profile(i_pc, factor, False)
                ax[i_pc, i_col].set_xlim(x_lim_profiles)
                ax[i_pc, i_col].set_title("Mean{:+.2f}$\cdot$PC{}".format(factor, i_pc+1))

            # Plot profiles.
            ax[i_pc, i_col].plot(v, altitudes, label="Parallel", color='#ff7f0e')

            if i_col > 0:  # For columns other than PC column, also plot magnitude line.
                marker_counter += 1
                ax[i_pc, i_col].plot(0.1, 0.1, 's', mfc="white", alpha=1, ms=12, mec='k', transform=ax[i_pc, i_col].transAxes)
                ax[i_pc, i_col].plot(0.1, 0.1, marker='${}$'.format(marker_counter), alpha=1, ms=7, mec='k', transform=ax[i_pc, i_col].transAxes)
            ax[i_pc, i_col].grid(True)

    # Add labels on x-axes.
    for i_col in range(shape_map[1]):
        if i_col == 0:
            ax[-1, i_col].set_xlabel("Coefficient of PC [-]")
        else:
            ax[-1, i_col].set_xlabel(x_label)


def plot_frequency_projection(x, y, labels=['PC1', 'PC2'], kde_plot=False, ax=None):
    if ax is None:
        ax = plt.figure(figsize=(5, 2.5)).gca()
        plt.subplots_adjust(top=0.975, bottom=0.178, left=0.15, right=0.94)

    if not kde_plot:
        # Create color map that yields more contrast for lower values than the baseline colormap.
        cmap_baseline = plt.get_cmap('bone_r')
        frac = .7
        clrs_low_values = cmap_baseline(np.linspace(0., .5, int(256*frac)))
        clrs_low_values[0, :] = 0.
        clrs_high_values = cmap_baseline(np.linspace(.5, 1., int(256*(1-frac))))
        cmap = ListedColormap(np.vstack((clrs_low_values, clrs_high_values)))

        n_bins = 120
        vmax = 1200
        h, _, _, im = ax.hist2d(x, y, bins=n_bins, cmap=cmap, norm=LogNorm(vmin=1, vmax=vmax))
        h_max = np.amax(h)
        print("Max occurences in hist2d bin:", str(h_max))
        if vmax is not None and h_max > vmax:
            print("Higher density occurring than anticipated.")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Occurrences [-]")
    else:
        import seaborn as sns
        sns.kdeplot(x=x, y=y, ax=ax)  #, bw_adjust=.6) # levels=lvls, cmap=cmap_grey, ax=ax_col[0])

    # plt.xlim(xlim_pc12)
    # plt.ylim(ylim_pc12)
    ax.grid(True)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    return ax


def run_pca(training_data):
    # Perform principal component analysis.
    n_features = training_data.shape[1]
    pca = PCA(n_components=n_features)
    pipeline = pca

    # print("{} features reduced to {} components.".format(n_features, n_components))
    pipeline.fit(training_data)
    print("{:.1f}% of variance retained using first two principal components.".format(np.sum(pca.explained_variance_ratio_[:2])*100))
    cum_var_exp = list(accumulate(pca.explained_variance_ratio_*100))
    print("Cumulative variance retained: " + ", ".join(["{:.2f}".format(var) for var in cum_var_exp]))

    mean_profile = pipeline.inverse_transform(np.zeros(n_features))
    for i_pc in range(2):
        profile_cmp = np.zeros(n_features)
        profile_cmp[i_pc] = 1
        print("PC{}".format(i_pc+1), pipeline.inverse_transform(profile_cmp)-mean_profile)

    return pipeline


def plot_pca_results(wind_data, pipeline, rl=.1):
    altitudes = wind_data['altitude']
    n_features = len(altitudes)
    data_pc = pipeline.transform(wind_data['training_data'])

    v_log = log_law_wind_profile2(altitudes, rl, 1, altitudes[-1], 0)
    pc_logn = pipeline.transform(v_log.reshape((1, -1)))
    print("Neutral log profile {}".format(rl), pc_logn[0, :2])
    data_pc = data_pc - pc_logn
    var = pipeline.explained_variance_
    ax = plot_frequency_projection(data_pc[:, 0], data_pc[:, 1])

    obukhov_lengths_inv_mark = 1/np.array([-100, -350, 1e10, 350, 100])
    obukhov_lengths_inv = 1/np.array([-.1, -5, -100, -350, -3e3, 1e10, 5e3, 2e3, 1e3, 700, 500, 350, 200, 100])
    if rl == .1:
        obukhov_lengths_inv[0] = -1
    pc1_log, pc2_log = [], []
    for oli in obukhov_lengths_inv:
        ol = 1/oli
        v_log = log_law_wind_profile2(altitudes, rl, 1, altitudes[-1], ol)
        # ax_log[i].plot(v_log, altitudes, label=ol)
        pc_log = pipeline.transform(v_log.reshape((1, -1))) - pc_logn
        pc1_log.append(pc_log[0, 0])
        pc2_log.append(pc_log[0, 1])
        if oli in obukhov_lengths_inv_mark:
            ax.plot(*pc_log[0, :2], '.', color='C1')
    ax.plot(pc1_log, pc2_log, '-', color='C1')
    # ax_log[i].set_xlim([0, None])
    # ax_log[i].legend()

    def get_pc_profile(i_pc=-1, multiplier=1., plot_pc=False):
        # Determine profile data by transforming data in PC to original coordinate system.
        if i_pc == -1:
            mean_profile = pipeline.inverse_transform(np.zeros(n_features))
            profile = mean_profile
        else:
            profile_cmp = np.zeros(n_features)
            profile_cmp[i_pc] = multiplier
            profile = pipeline.inverse_transform(profile_cmp)
            if plot_pc:
                mean_profile = pipeline.inverse_transform(np.zeros(n_features))
                profile -= mean_profile
        return profile

    plot_mean_and_pc_profiles(altitudes, var, get_pc_profile)

    return ax


def plot_profiles(pc12_profiles, altitudes, pipeline):
    n_features = len(altitudes)
    plt.figure()
    for i, pc12 in enumerate(pc12_profiles):
        pcs_profile = np.zeros(n_features)
        pcs_profile[:2] = pc12
        profile = np.insert(pipeline.inverse_transform(pcs_profile), 0, 0)
        plt.plot(profile, np.insert(altitudes, 0, 0), label=pc12)
    plt.xlim([0, 1.2])
    plt.legend()


def eval_loc(loc='mmc'):
    from read_data.dowa import read_data
    from preprocess_data import preprocess_data
    wind_data = read_data({'name': loc})
    wind_data = preprocess_data(wind_data)

    training_data = wind_data['training_data']
    pipeline = run_pca(training_data)
    if loc == 'mmc':
        rl = .1
    else:
        rl = .0002
    ax = plot_pca_results(wind_data, pipeline, rl)

    if loc == 'mmc':
        custom_cluster_profiles = np.array([
            [0, 0],
            [-.6, .27],
            [.15, .27],
            [.38, .02],
            [-.2, -.23]
        ])
        print(custom_cluster_profiles-np.array([-0.23897536, 0.19147062]))
    else:
        custom_cluster_profiles = np.array([
            [0, 0],
            [-.3, .11],
            [.25, -.08],
            [0., .15],
            [-.38, .13]
        ])
        print(custom_cluster_profiles-np.array([-0.574559, 0.35332352]))
    plot_profiles(custom_cluster_profiles, wind_data['altitude'], pipeline)

    n_features = len(wind_data['altitude'])
    for i, prfl in enumerate(custom_cluster_profiles):
        pcs_profile = np.zeros(n_features)
        pcs_profile[:2] = prfl
        print(i, pipeline.inverse_transform(pcs_profile))


if __name__ == '__main__':
    # from read_data.dowa import read_data
    # from preprocess_data import preprocess_data
    # wind_data_mmc = read_data({'name': 'mmc'})
    # wind_data_mmc = preprocess_data(wind_data_mmc)
    # wind_data_mmij = read_data({'name': 'mmij'})
    # wind_data_mmij = preprocess_data(wind_data_mmij)
    #
    # # training_data = np.vstack((wind_data_mmc['training_data']-np.mean(wind_data_mmc['training_data'], axis=0),
    # #                            wind_data_mmij['training_data']-np.mean(wind_data_mmij['training_data'], axis=0)))
    # training_data = wind_data_mmc['training_data']
    # pipeline = run_pca(training_data)
    # plot_pca_results(wind_data_mmc, pipeline)
    # # plot_pca_results(wind_data_mmij, pipeline)
    eval_loc('mmij')
    plt.show()
