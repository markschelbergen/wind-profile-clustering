import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from scipy.stats import gaussian_kde


def log_law_wind_profile2(z, z_0, v_ref, z_ref, ol):
    beta = 6
    gamma = 19.3

    if ol < 0:
        def psi(z, ol):
            x = (1 - gamma*z/ol)**.25
            psi = 2 * np.log((1+x)/2) + np.log((1+x**2)/2) - 2 * np.arctan(x) + np.pi/2
            return psi
        v = (np.log(z/z_0) - psi(z, ol))/(np.log(z_ref/z_0) - psi(z_ref, ol)) * v_ref
    elif ol == 0:
        v = np.log(z/z_0)/np.log(z_ref/z_0) * v_ref
    else:
        def psi(z, ol):
            psi = - beta * z/ol
            return psi
        v = (np.log(z/z_0) - psi(z, ol))/(np.log(z_ref/z_0) - psi(z_ref, ol)) * v_ref
    return v


def fit_err(x, wind_speed, profile_shapes):
    v_err = wind_speed - np.dot(x.reshape((1, -1)), profile_shapes)
    v_err = v_err.reshape(-1)
    return v_err


def eval_loc(loc='mmij'):
    from read_data.dowa import read_data
    wind_data = read_data({'name': loc})
    wind_speed = (wind_data['wind_speed_east']**2 + wind_data['wind_speed_north']**2)**.5
    n_samples = wind_speed.shape[0]

    if loc == 'mmc':
        rl = .1
    else:
        rl = .0002
    pcs = np.load("pcs_{}.npy".format(loc))
    shape_modes = np.insert(pcs, 0, log_law_wind_profile2(wind_data['altitude'], rl, 1, 200, 0), axis=0)

    x_sol = np.empty((n_samples, 3))
    for i in range(n_samples):
        v = wind_speed[i:i+1, :]
        x0 = np.array([
            np.amax(v),
            0,
            0
        ])
        x_sol[i, :] = least_squares(fit_err, x0, args=(v, shape_modes), bounds=((0, -np.inf, -np.inf), np.inf)).x
        print("{}/{}: {}".format(i, n_samples, x_sol[i, :]))
    np.save("x_sol_{}.npy".format(loc), x_sol)


def fit_err_cluster(x, wind_speed, profile_shape):
    v_err = wind_speed - x*profile_shape
    return v_err


def assign_to_cluster(loc):
    if loc == 'mmc':
        rl = .1
    else:
        rl = .0002

    from read_data.dowa import read_data
    wind_data = read_data({'name': loc})
    wind_speed = (wind_data['wind_speed_east'] ** 2 + wind_data['wind_speed_north'] ** 2) ** .5
    n_samples = wind_speed.shape[0]

    hand_picked_shapes = np.load("hand_picked_shapes_{}.npy".format(loc))
    shapes = np.insert(hand_picked_shapes, 0, log_law_wind_profile2(wind_data['altitude'], rl, 1, 200, 0), axis=0)
    n_curves = shapes.shape[0]

    x_cluster = np.empty((n_samples, 2))
    for i in range(n_samples):
        print("{}/{}".format(i, n_samples))
        v = wind_speed[i, :]
        v_norm = np.linalg.norm(v)
        costs = np.empty(n_curves)
        solutions = np.empty(n_curves)
        for j in range(n_curves):
            res = least_squares(fit_err_cluster, v_norm, args=(v, shapes[j, :]), bounds=(0, np.inf))
            costs[j] = res.cost
            solutions[j] = res.x
        i_min = np.argmin(costs)
        x_cluster[i, 0] = i_min
        x_cluster[i, 1] = solutions[i_min]
    np.save("x_cluster_{}.npy".format(loc), x_cluster)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_results(loc='mmij'):
    x_sol = np.load("x_sol_{}.npy".format(loc))
    x_cluster = np.load("x_cluster_{}.npy".format(loc))
    hand_picked_shapes = np.load("hand_picked_shapes_{}.npy".format(loc))

    altitudes = np.array([10., 20., 40., 60., 80., 100., 120., 140., 150., 160., 180., 200., 220., 250., 300., 500., 600.])
    if loc == 'mmc':
        rl = .1
    else:
        rl = .0002
    pcs = np.load("pcs_{}.npy".format(loc))
    shape_modes = np.insert(pcs, 0, log_law_wind_profile2(altitudes, rl, 1, 200, 0), axis=0)
    x_pc_plane = log_law_wind_profile2(100, rl, 1, 600, 0)

    if loc == 'mmc':
        hand_picked_shapes_pc = np.array([
            [-0.39793837, -0.07590686],
            [0.35206163, -0.07590686],
            [0.58206163, 0.17409314],
            [0.00206163, 0.42409314]
        ])
    else:
        hand_picked_shapes_pc = np.array([
            [-0.42416149, -0.05207391],
            [0.27583851, -0.02207391],
            [-0.12416149, 0.20792609],
            [-0.50416149, 0.11792609]
        ])

    # ax1 = plt.figure().gca()
    # ax1.set_xlabel('Log profile [m/s]')
    # ax1.set_ylabel('Scaling PC1 [m/s]')
    # sns.kdeplot(x=x_sol[:, 0], y=x_sol[:, 1], bw_adjust=.6, cmap=truncate_colormap(plt.get_cmap("Blues"), 0.2, 1.))
    # ax2 = plt.figure().gca()
    # ax2.set_xlabel('Log profile [m/s]')
    # ax2.set_ylabel('Scaling PC2 [m/s]')
    # sns.kdeplot(x=x_sol[:, 0], y=x_sol[:, 2], bw_adjust=.6, cmap=truncate_colormap(plt.get_cmap("Blues"), 0.2, 1.))
    # ax1.axvline(x_pc_plane, color='grey')
    # ax2.axvline(x_pc_plane, color='grey')
    # for i, pc_prfl in enumerate(hand_picked_shapes_pc):
    #     ax1.plot(x_pc_plane, pc_prfl[0], 's')
    #     ax2.plot(x_pc_plane, pc_prfl[1], 's')

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # subset = np.random.choice(x_sol.shape[0], x_sol.shape[0] // 10, replace=False)
    # ax.scatter(x_sol[subset, 0], x_sol[subset, 1], x_sol[subset, 2], c=x_cluster[subset, 0], alpha=.02)
    # ax.set_xlabel('Magnitude')

    fig = plt.figure()
    ax3d = fig.add_subplot(projection='3d')
    ax3d.set_xlabel('$v_{log,200m}$ [m/s]')
    ax3d.set_xlim([0, 30])
    ax3d.set_ylabel('$k_{PC1}$ [-]')
    ax3d.set_zlabel('$k_{PC2}$ [-]')

    for prfl in hand_picked_shapes:
        hand_picked_shape_lines = np.zeros((2, 3))
        r = 25
        v_prfl = prfl*r
        x0 = np.array([
            np.amax(v_prfl),
            0,
            0
        ])
        hand_picked_shape_lines[1, :] = least_squares(fit_err, x0, args=(v_prfl, shape_modes), bounds=((0, -np.inf, -np.inf), np.inf)).x
        ax3d.plot(hand_picked_shape_lines[:, 0], hand_picked_shape_lines[:, 1], hand_picked_shape_lines[:, 2])
    ax3d.plot([0, r], [0, 0], [0, 0], 'k')

    n_bins = 6
    bin_edges = np.linspace(0, 26, n_bins+1)
    # for le, ue in zip(bin_edges[:-1], bin_edges[1:]):

    n_total = x_sol.shape[0]
    ax = plt.subplots(2, n_bins//2, sharex=True, sharey=True)[1].reshape(-1)

    for i, (le, ue) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (x_sol[:, 0] >= le) & (x_sol[:, 0] < ue)
        n_bin = np.sum(mask)
        print("Percentage of points in bins: {:.1f}%".format(n_bin/n_total*100))
        kernel = gaussian_kde(x_sol[mask, 1:].T)

        x, y = np.mgrid[-20:20:50j, -10:10:50j]
        positions = np.vstack([x.ravel(), y.ravel()])
        z = np.reshape(kernel(positions).T, x.shape)*n_bin/n_total*100

        print(np.amax(z))

        clevels = np.array([0.03, 0.05, 0.1, 0.25, 0.5, 0.75, 1., 2.])*3/n_bins  #, 4., 6., 8.
        print(clevels)


        # ax = plt.figure().gca()
        # cp = ax.contour(x, y, z, cmap=plt.cm.gist_earth_r)
        # print(cp.cvalues)
        # ax.clabel(cp, inline=True, fontsize=10)

        # ax[0].imshow(np.rot90(z), cmap=plt.cm.gist_earth_r, extent=[-20, 20, -10, 10])
        cset = ax[i].contour(x, y, z, clevels, cmap=truncate_colormap(plt.cm.gist_earth_r, 0.05, 1.), norm=colors.LogNorm())  #, (le+ue)/2, zdir='x'
        ax[i].clabel(cset, inline=True, fontsize=10)
        # sns.kdeplot(x=x_sol[:, 1], y=x_sol[:, 2], bw_adjust=.6, cmap=truncate_colormap(plt.get_cmap("Blues"), 0.2, 1.), ax=ax[2])

        # clevels = np.array([1, 5, 10])/n_bins

        ax3d.contour(z, x, y, clevels, offset=(le+ue)/2, zdir='x', cmap=truncate_colormap(plt.cm.gist_earth_r, 0.05, 1.), norm=colors.LogNorm())



def plot_distr_pc12(loc='mmij'):
    x_sol = np.load("x_sol_{}.npy".format(loc))
    sns.kdeplot(x=x_sol[:, 1], y=x_sol[:, 2], bw_adjust=.6, cmap=truncate_colormap(plt.get_cmap("Blues"), 0.2, 1.)) # levels=lvls, cmap=cmap_grey, ax=ax_col[0])


def plot_results_pc1(loc='mmij'):
    x_sol = np.load("x_sol_{}.npy".format(loc))
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    sns.kdeplot(x=x_sol[:, 0], y=x_sol[:, 1], bw_adjust=.6, cmap=truncate_colormap(plt.get_cmap("Blues"), 0.2, 1.), ax=ax[0, 0]) # levels=lvls, cmap=cmap_grey, ax=ax_col[0])
    mask = x_sol[:, 2] > 1
    sns.kdeplot(x=x_sol[mask, 0], y=x_sol[mask, 1], bw_adjust=.6, cmap=truncate_colormap(plt.get_cmap("Oranges"), 0.2, 1.), ax=ax[1, 0]) # levels=lvls, cmap=cmap_grey, ax=ax_col[0])
    sns.kdeplot(x=x_sol[~mask, 0], y=x_sol[~mask, 1], bw_adjust=.6, cmap=truncate_colormap(plt.get_cmap("Oranges"), 0.2, 1.), ax=ax[1, 1]) # levels=lvls, cmap=cmap_grey, ax=ax_col[0])
    # plt.xlim([0, 27])
    # plt.ylim([-5, 8])


def plot_results_pc2(loc='mmij'):
    x_sol = np.load("x_sol_{}.npy".format(loc))
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    sns.kdeplot(x=x_sol[:, 0], y=x_sol[:, 2], bw_adjust=.6, cmap=truncate_colormap(plt.get_cmap("Blues"), 0.2, 1.), ax=ax[0, 0]) # levels=lvls, cmap=cmap_grey, ax=ax_col[0])
    mask = x_sol[:, 1] > 0
    sns.kdeplot(x=x_sol[mask, 0], y=x_sol[mask, 2], bw_adjust=.6, cmap=truncate_colormap(plt.get_cmap("Oranges"), 0.2, 1.), ax=ax[1, 0]) # levels=lvls, cmap=cmap_grey, ax=ax_col[0])
    sns.kdeplot(x=x_sol[~mask, 0], y=x_sol[~mask, 2], bw_adjust=.6, cmap=truncate_colormap(plt.get_cmap("Oranges"), 0.2, 1.), ax=ax[1, 1]) # levels=lvls, cmap=cmap_grey, ax=ax_col[0])
    # plt.xlim([0, 27])
    # plt.ylim([-5, 8])


if __name__ == '__main__':
    loc = 'mmc'
    # assign_to_cluster(loc)
    # eval_loc(loc)
    plot_results(loc)
    # plot_distr_pc12()
    # plot_results_pc1(loc)
    # plot_results_pc2(loc)
    plt.show()