from sklearn.decomposition import PCA
from itertools import accumulate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from dimensionality_reduction import *
from scipy.optimize import least_squares
from scipy.interpolate import interp1d, Rbf, UnivariateSpline, Akima1DInterpolator
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit, minimize
from utils import add_panel_labels


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
        try:
            a.text(.05, y[i], label, transform=a.transAxes, fontsize='large')  #, fontweight='bold', va='top', ha='right')
        except:
            a.text(-10, -25, 0, label, fontsize='large')


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

    ax.set_xlim([-1, 2])
    ax.set_ylim([-.25, .85])
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
    if loc == 'mmca':
        rl = .1
    else:
        rl = .0002
    print("Roughness length {}".format(rl))
    altitudes = wind_data['altitude']
    n_features = len(altitudes)
    data_pc = pipeline.transform(wind_data['training_data'])

    vlog_max = 1

    v_logn = log_law_wind_profile2(altitudes, rl, vlog_max, 600, 0)
    pc_logn = pipeline.transform(v_logn.reshape((1, -1)))
    if transform_pcs and loc == 'mmca':
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
        if loc == 'mmca':
            data_pc[:, 1] = -data_pc[:, 1]
        data_pc = data_pc - pc_logn
    var = pipeline.explained_variance_
    ax_2dhist = plot_frequency_projection(data_pc[:, 0], data_pc[:, 1])

    obukhov_lengths_inv_mark = []  #1/np.array([-100, -350, 1e10, 350, 100])
    obukhov_lengths_inv = 1/np.array([-100, -350, -3e3, 1e10, 5e3, 2e3, 1e3, 700, 500, 350, 200, 100])
    if rl == .1:
        obukhov_lengths_inv[0] = -1
    pc1_log, pc2_log = [], []
    # pc1_log_alt, pc2_log_alt = [], []

    if obukhov_lengths_inv_mark:
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
            if loc == 'mmca':
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
    if obukhov_lengths_inv_mark:
        ax_prfl.legend()
    ax_2dhist.plot(pc1_log, pc2_log, '-', color='salmon', linewidth=1, label='Logarithmic profiles')
    # ax_2dhist.plot(pc1_log_alt, pc2_log_alt, '-', color='C2')
    if not transform_pcs:
        ax_2dhist.plot(*pc_logn[0, :2], '*', ms=8, mfc='None', color='k')
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
            if loc == 'mmca':
                profile_cmp[1] = -profile_cmp[1]
            profile = pipeline.inverse_transform(profile_cmp)
            if plot_pc:
                mean_profile = pipeline.inverse_transform(np.zeros(n_features))
                profile -= mean_profile
        return profile

    plot_mean_and_pc_profiles(altitudes, var, get_pc_profile, rl=rl)

    use_hp_shapes = False
    if use_hp_shapes:
        if loc == 'mmca':
            hand_picked_shapes_pc = np.array([
                [-.4, -.08],
                [-.13, .47],
                [.35, .32],
                [.65, .1],
            ])
            hand_picked_shapes_pc = hand_picked_shapes_pc+pc_logn[0, :2]
            hand_picked_shapes_pc[:, 1] = -hand_picked_shapes_pc[:, 1]
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
                if loc == 'mmca':
                    pc_prfl[1] = -pc_prfl[1]
                pc_prfl = pc_prfl - pc_logn[0, :2]
            ax_2dhist.plot(*pc_prfl, '*', ms=8, mfc='None', color='C{}'.format(i))

            # x = least_squares(fit_err, [1, 0, 0], args=(hand_picked_shapes[i, :], shape_modes), bounds=((0, -np.inf, -np.inf), np.inf)).x
            # ax_2dhist.plot(*x[1:], 'x', mfc='None', color='C2')
        ax_2dhist.plot(0, 0, '*', ms=8, mfc='None', color='k', label='Hand-picked shapes')

        np.save("hand_picked_shapes_{}.npy".format(loc), hand_picked_shapes)
        plot_wind_profile_shapes(hand_picked_shapes, wind_data['altitude'], rl, labels=hand_picked_shapes_pc - pc_logn[0, :2])

    ax_2dhist.legend(bbox_to_anchor=(0.05, 1.05, .9, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)

    return ax_2dhist, pc_logn


def plot_wind_profile_shapes(profiles, altitudes, roughness_length, labels=None, colors=None, ax=None):
    if ax is None:
        figsize = (3, 3)
        plt.figure(figsize=figsize)
        plt.subplots_adjust(top=0.95, bottom=0.155, left=0.2, right=0.97, hspace=0.2, wspace=0.2)
        ax = plt.gca()
        ax.set_ylabel('Height [m]')
    alts = np.linspace(0, altitudes[-1], 100)
    for i, v in enumerate(profiles):
        if isinstance(labels, list):
            lbl = labels[i]
        else:
            lbl = '({:.2f}, {:.2f})'.format(*labels[i, :])
        if colors is None:
            clr = 'C{}'.format(i)
        else:
            clr = colors[i]

        # ax.plot(v, altitudes, '*', label=lbl, color='C{}'.format(i))
        # alts2 = np.linspace(altitudes[0], altitudes[-1], 1000)
        # f = interp1d(np.insert(altitudes[4:], 0, 0), np.insert(v[4:], 0, 0), kind='quadratic', fill_value='extrapolate')
        # ax.plot(f(alts2), alts2, color='C{}'.format(i))

        f = Akima1DInterpolator(np.insert(altitudes[1:], 0, 0), np.insert(v[1:], 0, 0))
        ax.plot(f(alts), alts, color=clr, label=lbl)
    alts[0] = 1e-3
    ax.plot(log_law_wind_profile2(alts, roughness_length, 1., altitudes[-1], 0.), alts, ':', color='k', label='Log')

    ax.set_xlim([0, 1.2])
    ax.set_ylim([0, 600])
    ax.set_xlabel(r"$\tilde{v}$ [-]")
    ax.grid()
    ax.legend()


def eval_loc(loc='mmca', transform_pcs=True):
    from read_data.dowa import read_data
    from preprocess_data import preprocess_data
    wind_data = read_data({'name': loc})
    wind_data_red = preprocess_data(wind_data)
    wind_data_full = preprocess_data(wind_data, False)

    pipeline, pcs = run_pca(wind_data_red['training_data'], wind_data_red['altitude'])
    if loc == 'mmca':
        pcs[1, :] = -pcs[1, :]
    np.save("pcs_{}.npy".format(loc), pcs)

    ax, pc_logn = plot_pca_results(wind_data_red, pipeline, loc, pcs, transform_pcs)

    data_pc = pipeline.transform(wind_data_full['training_data'])
    if transform_pcs:
        if loc == 'mmca':
            data_pc[:, 1] = -data_pc[:, 1]
        data_pc = data_pc - pc_logn
        np.save('pc_data_{}.npy'.format(loc), data_pc)

    data_pc = pipeline.transform(wind_data_red['training_data'])
    if transform_pcs:
        if loc == 'mmca':
            data_pc[:, 1] = -data_pc[:, 1]
        data_pc = data_pc - pc_logn
    if loc == 'mmij':
        allocate_clusters_mmij(data_pc, pipeline, ax, wind_data['altitude'], pc_logn[0, :2])
    else:
        allocate_clusters_mmca(data_pc, pipeline, ax, wind_data['altitude'], pc_logn[0, :2])


def allocate_clusters_mmca(data_pc, pipeline, ax, altitudes, pc_logn):
    rl = .1
    from matplotlib.patches import Polygon, Path
    cluster0 = np.array([[-.733, -.185],
                         [.173, -.185],
                         [.173, 0],
                         [-.733, 0]])
    cluster1 = np.array([cluster0[1, :],
                         [cluster0[1, 0], -.05],
                         [1.3, 0.0],
                         [1.3, cluster0[1, 1]]])
    cluster2 = np.array([cluster1[1, :],
                         cluster0[2, :],
                         [1.3, .4],
                         cluster1[2, :]])
    cluster3 = np.array([cluster0[2, :],
                       [-.055, .147],
                       [.5, .8],
                       cluster2[2, :]])
    cluster4 = np.array([cluster3[1, :],
                         [-.55, cluster3[1, 1]],
                         [-.55, .8],
                         cluster3[2, :]])

    mean_profiles_pc = []
    mean_profiles = []
    frequency = []
    for i, corners in enumerate([cluster0, cluster1, cluster2, cluster3, cluster4]):
        corners = corners
        path = Path(corners)
        mask = path.contains_points(data_pc[:, :2])
        frequency.append(np.sum(mask)/data_pc.shape[0]*100)
        mean_prfl_pc = np.mean(data_pc[mask, :], axis=0)
        mean_profiles_pc.append(mean_prfl_pc[:5])
        if i == 0:
            lbl = 'Cluster mean'
        else:
            lbl = None
        ax.plot(*mean_prfl_pc[:2], '*', ms=8, mfc='None', color='C{}'.format(i), label=lbl)

        mean_prfl_pc[:2] = mean_prfl_pc[:2] + pc_logn
        mean_prfl_pc[1] = -mean_prfl_pc[1]

        mean_profiles.append(pipeline.inverse_transform(mean_prfl_pc))

        poly = Polygon(corners, edgecolor='grey', linewidth=.7, facecolor='none')  #'C{}'.format(i)
        ax.add_patch(poly)
    mean_profiles = np.vstack(mean_profiles)
    mean_profiles_pc = np.vstack(mean_profiles_pc)

    ax.plot(0, 0, '*', ms=8, mfc='None', color='k')
    ax.plot(*-pc_logn, 'o', color='k', ms=5, mfc='None', label='Mean')
    ax.legend(bbox_to_anchor=(0.05, 1.05, .9, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)

    fig, ax_pcs = plt.subplots(5, 1, sharex=True)
    plt.suptitle("PC coefficients op clusters")
    for i, ax in enumerate(ax_pcs):
        ax.bar(range(5), mean_profiles_pc[:, i])
        ax.set_ylabel('PC{}'.format(i+1))

    plt.figure()
    plt.bar(range(1, 6), frequency, color=['C{}'.format(i) for i in range(5)])

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(3, 3))
    plt.subplots_adjust(top=0.96, bottom=0.155, left=0.21, right=0.98, hspace=0.2, wspace=0.2)
    plot_wind_profile_shapes(mean_profiles, altitudes, rl,
                             labels=['Cluster {}'.format(i+1) for i in range(len(mean_profiles))],
                             colors=['C{}'.format(i) for i in range(len(mean_profiles))], ax=ax)
    ax.set_ylabel('Height [m]')

    np.save("cluster_shapes_mmca.npy", mean_profiles)


def allocate_clusters_mmij(data_pc, pipeline, ax, altitudes, pc_logn):
    rl = .0002
    from matplotlib.patches import Polygon, Path
    baseline = np.array([[-.447, -.039],
                         [1.7, .08]])
    underline = np.array([[-.447, -.155],
                          [1.7, -.12]])
    cluster0 = np.array([baseline[0, :],
                         [.07, np.interp(.07, baseline[:, 0], baseline[:, 1])],
                         [.07, np.interp(.07, underline[:, 0], underline[:, 1])],
                         underline[0, :]])
    cluster1 = np.array([cluster0[2, :],
                         cluster0[1, :],
                         [.7, np.interp(.7, baseline[:, 0], baseline[:, 1])],
                         [.7, np.interp(.7, underline[:, 0], underline[:, 1])]])
    cluster2 = np.array([cluster1[3, :],
                         cluster1[2, :],
                         baseline[1, :],
                         underline[1, :]])
    cluster3 = np.array([[-.247, .43],
                       [-.203, np.interp(-.203, baseline[:, 0], baseline[:, 1])],
                       baseline[0, :],
                       [-.447, .43]])
    cluster4 = np.array([cluster3[1, :],
                         [.7, np.interp(.7, baseline[:, 0], baseline[:, 1])],
                         [.3, .3],
                         [np.interp(.3, cluster3[1::-1, 1], cluster3[1::-1, 0]), .3]])
    cluster5 = np.array([cluster4[3, :],
                         cluster3[0, :],
                         [-.213, .629],
                         [.243, .629],
                         cluster4[2, :]])

    mean_profiles_pc = []
    mean_profiles = []
    frequency = []
    for i, corners in enumerate([cluster0, cluster1, cluster2, cluster3, cluster4, cluster5]):
        corners = corners - pc_logn
        path = Path(corners)
        mask = path.contains_points(data_pc[:, :2])
        frequency.append(np.sum(mask)/data_pc.shape[0]*100)
        mean_prfl_pc = np.mean(data_pc[mask, :], axis=0)
        mean_profiles_pc.append(mean_prfl_pc[:5])
        ax.plot(*mean_prfl_pc[:2], '*', ms=8, mfc='None', color='C{}'.format(i))

        mean_prfl_pc[:2] = mean_prfl_pc[:2] + pc_logn

        mean_profiles.append(pipeline.inverse_transform(mean_prfl_pc))

        poly = Polygon(corners, edgecolor='grey', linewidth=.5, facecolor='none')  #'C{}'.format(i)
        ax.add_patch(poly)
    mean_profiles = np.vstack(mean_profiles)
    mean_profiles_pc = np.vstack(mean_profiles_pc)

    fig, ax_pcs = plt.subplots(5, 1, sharex=True)
    plt.suptitle("PC coefficients op clusters")
    for i, ax in enumerate(ax_pcs):
        ax.bar(range(6), mean_profiles_pc[:, i])
        ax.set_ylabel('PC{}'.format(i + 1))

    plt.figure()
    plt.bar(range(1, 7), frequency, color=['C{}'.format(i) for i in range(6)])

    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(5, 3))
    plt.subplots_adjust(top=0.96, bottom=0.155, left=0.175, right=0.995, hspace=0.2, wspace=0.2)
    plot_wind_profile_shapes(mean_profiles[:3], altitudes, rl,
                             labels=['Cluster {}'.format(i+1) for i in range(3)],
                             colors=['C{}'.format(i) for i in range(3)], ax=ax[0])
    plot_wind_profile_shapes(mean_profiles[3:], altitudes, rl,
                             labels=['Cluster {}'.format(i+1) for i in range(3, 6)],
                             colors=['C{}'.format(i) for i in range(3, 6)], ax=ax[1])
    ax[1].legend(loc=6)
    add_panel_labels(ax, [.45, .15])
    ax[0].set_ylabel('Height [m]')

    np.save("cluster_shapes_mmij.npy", mean_profiles)


if __name__ == '__main__':
    # from read_data.dowa import read_data
    # from preprocess_data import preprocess_data
    # wind_data_mmca = read_data({'name': 'mmca'})
    # wind_data_mmca = preprocess_data(wind_data_mmca)
    # wind_data_mmij = read_data({'name': 'mmij'})
    # wind_data_mmij = preprocess_data(wind_data_mmij)
    #
    # # training_data = np.vstack((wind_data_mmca['training_data']-np.mean(wind_data_mmca['training_data'], axis=0),
    # #                            wind_data_mmij['training_data']-np.mean(wind_data_mmij['training_data'], axis=0)))
    # training_data = wind_data_mmca['training_data']
    # pipeline = run_pca(training_data)
    # plot_pca_results(wind_data_mmca, pipeline)
    # # plot_pca_results(wind_data_mmij, pipeline)
    eval_loc('mmca')
    plt.show()
