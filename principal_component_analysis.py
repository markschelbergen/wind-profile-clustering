from sklearn.decomposition import PCA
from itertools import accumulate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from wind_resource_fit import log_law_wind_profile2, fit_err
from scipy.optimize import least_squares
from scipy.interpolate import interp1d, Rbf, UnivariateSpline, Akima1DInterpolator
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit, minimize

xlim_pc12 = [-1.1, 1.1]
ylim_pc12 = [-1.1, 1.1]
x_lim_profiles = [0., 1.25]

def add_inside_panel_labels(ax, y=None):
    import string
    ax = ax.reshape(-1)
    if y is None:
        y = [.85]*ax.shape[0]
    for i, a in enumerate(ax):
        label = '('+string.ascii_lowercase[i]+')'
        a.text(.05, y[i], label, transform=a.transAxes, fontsize='large')  #, fontweight='bold', va='top', ha='right')


def plot_mean_and_pc_profiles(altitudes, var, get_profile, rl=.1):
    plot_n_pcs = 2

    x_label = r"$\tilde{v}$ [-]"
    figsize = (6, 3)

    fig, ax = plt.subplots(1, 3, sharey=True, figsize=figsize)
    wspace = 0.1
    layout = {'top': 0.9, 'bottom': 0.17, 'left': 0.1, 'right': 0.985, 'hspace': 0.23}
    plt.subplots_adjust(**layout, wspace=wspace)

    alts = np.linspace(1e-3, altitudes[-1], 100)
    ax[0].plot(log_law_wind_profile2(alts, rl, 1, altitudes[-1], 0), alts, label='Log')
    v = get_profile()
    f = Akima1DInterpolator(np.insert(altitudes[1:], 0, 0), np.insert(v[1:], 0, 0))
    ax[0].plot(f(alts), alts, 'k:', label="Mean")

    ax[0].legend()
    ax[0].set_title("Origin")
    ax[0].grid(True)
    ax[0].set_ylabel("Height [m]")
    ax[0].set_xlim(x_lim_profiles)
    ax[0].set_xlabel(x_label)


    # Plot PCs and PC multiplicands superimposed on the mean.
    for i_pc in range(plot_n_pcs):  # For every PC/row in the plot.
        # Get profile data.
        v = get_profile(i_pc, 1, True)
        ax[i_pc+1].set_xlim(xlim_pc12)
        ax[i_pc+1].set_title("PC{}".format(i_pc+1))

        # Plot profiles.
        ax[i_pc+1].plot(v, altitudes)
        ax[i_pc+1].grid(True)
        ax[i_pc+1].set_xlabel("Coefficient of PC [-]")

    add_inside_panel_labels(ax, [.68, .85, .85])


def plot_frequency_projection(x, y, labels=['PC1', 'PC2'], kde_plot=False, ax=None):
    if ax is None:
        ax = plt.figure(figsize=(5, 3)).gca()
        plt.subplots_adjust(top=0.81, bottom=0.178, left=0.15, right=0.94)

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


def run_pca(training_data, altitudes):
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
    pcs = np.empty((2, n_features))
    for i_pc in range(2):
        profile_cmp = np.zeros(n_features)
        profile_cmp[i_pc] = 1
        pcs[i_pc, :] = pipeline.inverse_transform(profile_cmp)-mean_profile
        print("PC{}".format(i_pc+1), pcs[i_pc, :])
        print("Mean value: {:.2f}".format(np.trapz(pcs[i_pc, :], altitudes)/altitudes[-1]))

    return pipeline, pcs


def plot_pca_results(wind_data, pipeline, loc, pcs, transform_pcs=True):
    if loc == 'mmc':
        rl = .1
    else:
        rl = .0002
    altitudes = wind_data['altitude']
    n_features = len(altitudes)
    data_pc = pipeline.transform(wind_data['training_data'])

    vlog_max = 1

    v_logn = log_law_wind_profile2(altitudes, rl, vlog_max, 600, 0)
    pc_logn = pipeline.transform(v_logn.reshape((1, -1)))
    if transform_pcs:
        pc_logn[0, 1] = -pc_logn[0, 1]

    def alternative_transform(profile):
        shape_modes = np.insert(pcs, 0, log_law_wind_profile2(altitudes, rl, vlog_max, 600, 0), axis=0)
        x = least_squares(fit_err, [1, 0, 0], args=(profile, shape_modes),
                          bounds=((0, -np.inf, -np.inf), np.inf)).x
        return x

    # v_log_pc12 = pipeline.inverse_transform(np.pad(pc_logn[0, :2], (0, n_features - 2)))
    # x = alternative_transform(v_log_pc12)
    # pc_logn_correction = x[1:]
    # plt.figure()
    # plt.title("Dimensional reduction effect")
    # plt.plot(v_log, altitudes, label='log')
    # plt.plot(v_log_pc12, altitudes, label='log pc12')
    # shape_modes = np.insert(pcs, 0, log_law_wind_profile2(altitudes, rl, 1, 600, 0), axis=0)
    # print(pc_logn_correction, shape_modes.shape)
    # plt.plot(x@shape_modes, altitudes, 'k--', label='fit')
    # plt.legend()
    # pc_logn = pc_logn - np.pad(pc_logn_correction, (0, n_features - 2))
    # print("Neutral log profile {}".format(rl), pc_logn[0, :2])

    if transform_pcs:
        if loc == 'mmc':
            data_pc[:, 1] = -data_pc[:, 1]
        data_pc = data_pc - pc_logn
    var = pipeline.explained_variance_
    ax_2dhist = plot_frequency_projection(data_pc[:, 0], data_pc[:, 1])

    obukhov_lengths_inv_mark = 1/np.array([-100, -350, 1e10, 350, 100])
    obukhov_lengths_inv = 1/np.array([-.1, -5, -100, -350, -3e3, 1e10, 5e3, 2e3, 1e3, 700, 500, 350, 200, 100])
    if rl == .1:
        obukhov_lengths_inv[0] = -1
    pc1_log, pc2_log = [], []
    # pc1_log_alt, pc2_log_alt = [], []

    ax_prfl = plt.figure().gca()
    for i, oli in enumerate(obukhov_lengths_inv):
        ol = 1/oli
        if ol == 1e10:
            v_log = v_logn
        else:
            v_log = log_law_wind_profile2(altitudes, rl, vlog_max, altitudes[-1], ol)
        # ax_log[i].plot(v_log, altitudes, label=ol)

        pc_log = pipeline.transform(v_log.reshape((1, -1)))
        pc_logt = pc_log.copy()

        if transform_pcs:
            if loc == 'mmc':
                pc_logt[:, 1] = -pc_logt[:, 1]
            pc_logt = pc_logt - pc_logn

            # pc_log_alt = alternative_transform(v_log)
            # pc1_log_alt.append(pc_log_alt[1])
            # pc2_log_alt.append(pc_log_alt[2])

        pc1_log.append(pc_logt[0, 0])
        pc2_log.append(pc_logt[0, 1])
        if oli in obukhov_lengths_inv_mark:
            ax_2dhist.plot(*pc_logt[0, :2], '.', color='C0', ms=8)

            ax_prfl.plot(v_log, altitudes, color='C{}'.format(i), label='OL={:.1e} PCs=[{:.1f}, {:.1f}, {:.2f}]'.format(ol, *pc_log[0, :3]))
            pcn = 3
            v_log_pc1n = pipeline.inverse_transform(np.pad(pc_log[0, :pcn], (0, n_features - pcn)))
            ax_prfl.plot(v_log_pc1n, altitudes, '--', color='C{}'.format(i))
            # ax_prfl.plot(pc_log_alt@shape_modes, altitudes, ':', color='C{}'.format(i))
    ax_prfl.legend()
    ax_2dhist.plot(pc1_log, pc2_log, '-', color='C0', label='Logarithmic profiles')
    # ax_2dhist.plot(pc1_log_alt, pc2_log_alt, '-', color='C2')
    if not transform_pcs:
        ax_2dhist.plot(*pc_logn[0, :2], 'o', mfc='None', color='C1')
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
            if loc == 'mmc':
                profile_cmp[1] = -profile_cmp[1]
            profile = pipeline.inverse_transform(profile_cmp)
            if plot_pc:
                mean_profile = pipeline.inverse_transform(np.zeros(n_features))
                profile -= mean_profile
        return profile

    plot_mean_and_pc_profiles(altitudes, var, get_pc_profile)

    if loc == 'mmc':
        hand_picked_shapes_pc = np.array([
            [-.6, .27],
            [.15, .27],
            [.38, .02],
            [-.2, -.23]
        ])
    else:
        hand_picked_shapes_pc = np.array([
            [-.3, -.11],
            [.4, -.08],
            [0., .15],
            [-.38, .06]
        ])

    hand_picked_shapes = np.zeros((hand_picked_shapes_pc.shape[0], n_features))
    for i, pc_prfl in enumerate(hand_picked_shapes_pc):
        hand_picked_shapes[i, :] = pipeline.inverse_transform(np.pad(pc_prfl, (0, n_features - 2)))

        if transform_pcs:
            if loc == 'mmc':
                pc_prfl[1] = -pc_prfl[1]
            pc_prfl = pc_prfl - pc_logn[0, :2]
        ax_2dhist.plot(*pc_prfl, '*', ms=8, mfc='None', color='C1')

        # x = least_squares(fit_err, [1, 0, 0], args=(hand_picked_shapes[i, :], shape_modes), bounds=((0, -np.inf, -np.inf), np.inf)).x
        # ax_2dhist.plot(*x[1:], 'x', mfc='None', color='C2')
    ax_2dhist.plot(0, 0, '*', ms=8, mfc='None', color='C1', label='Hand-picked shapes')
    print(pc_logn[0, :2])
    ax_2dhist.plot(*-pc_logn[0, :2], 'o', color='k', ms=5, mfc='None', label='Mean')

    np.save("hand_picked_shapes_{}.npy".format(loc), hand_picked_shapes)
    plot_hand_picked_shapes(hand_picked_shapes, wind_data['altitude'], labels=hand_picked_shapes_pc-pc_logn[0, :2])
    print(wind_data['altitude'], hand_picked_shapes)

    ax_2dhist.set_xlim([-1, 2])
    ax_2dhist.set_ylim([-.25, .8])
    ax_2dhist.legend(bbox_to_anchor=(0.05, 1.05, .9, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)

    return ax_2dhist


def plot_hand_picked_shapes(profiles, altitudes, labels=None):
    figsize = (3, 3)
    plt.figure(figsize=figsize)
    plt.subplots_adjust(top=0.95, bottom=0.155, left=0.2, right=0.97, hspace=0.2, wspace=0.2)
    for i, v in enumerate(profiles):
        lbl = '({:.2f}, {:.2f})'.format(*labels[i, :])
        # plt.plot(v, altitudes, '*', label=lbl, color='C{}'.format(i))
        # alts = np.linspace(altitudes[0], altitudes[-1], 1000)
        # f = interp1d(np.insert(altitudes[4:], 0, 0), np.insert(v[4:], 0, 0), kind='quadratic', fill_value='extrapolate')
        # plt.plot(f(alts), alts, color='C{}'.format(i))

        alts = np.linspace(0, altitudes[-1], 100)
        f = Akima1DInterpolator(np.insert(altitudes[1:], 0, 0), np.insert(v[1:], 0, 0))
        plt.plot(f(alts), alts, color='C{}'.format(i), label=lbl)

    plt.xlim([0, 1.2])
    plt.ylim([0, 600])
    plt.xlabel(r"$\tilde{v}$ [-]")
    plt.ylabel('Height [m]')
    plt.grid()
    plt.legend()


def eval_loc(loc='mmc'):
    from read_data.dowa import read_data
    from preprocess_data import preprocess_data
    wind_data = read_data({'name': loc})
    wind_data = preprocess_data(wind_data)

    pipeline, pcs = run_pca(wind_data['training_data'], wind_data['altitude'])
    if loc == 'mmc':
        pcs[1, :] = -pcs[1, :]
    np.save("pcs_{}.npy".format(loc), pcs)
    ax = plot_pca_results(wind_data, pipeline, loc, pcs)


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
    eval_loc('mmc')
    plt.show()
