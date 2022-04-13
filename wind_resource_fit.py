import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns


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
        profile_shapes = np.array([
            log_law_wind_profile2(wind_data['altitude'], rl, 1, 100, 0),
            [-0.23049645, -0.24630588, -0.28568619, -0.2984786, -0.30250081, -0.30061507,
             -0.29401967, -0.28247869, -0.27554446, -0.26734227, -0.25023463, -0.2305409,
             -0.20981732, -0.17730358, -0.12295292, 0.05224662, 0.10445487],
            [0.53796995, 0.4677825, 0.29697192, 0.15289426, 0.05002982, -0.03115965,
             -0.09539336, -0.14123821, -0.15740775, -0.16916436, -0.18807282, -0.19300634,
             -0.18839821, -0.16899669, -0.10964258, 0.20027731, 0.32476833]
        ])
    else:
        rl = .0002
        profile_shapes = np.array([
            log_law_wind_profile2(wind_data['altitude'], rl, 1, 100, 0),
            # [0.39414811, 0.42479271, 0.50331044, 0.55509773, 0.59572789, 0.6302155,
            # 0.66023076, 0.68549557, 0.69744642, 0.70813617, 0.72941409, 0.74813871,
            # 0.76576865, 0.79007113, 0.82544999, 0.94181701, 0.98426081],
            # [0.5697327, 0.62800094, 0.77829325, 0.87917178, 0.94648236, 0.99033134,
            # 1.01496217, 1.01710714, 1.01456583, 1.00576285, 0.98689486, 0.95632611,
            # 0.92148368, 0.86155488, 0.74954797, 0.34853172, 0.22740257],
            [-0.3711465, -0.36080289, -0.33244151, -0.30885968, -0.28844681, -0.2697489,
             -0.25224161, -0.23609953, -0.22805606, -0.22041634, -0.20493771, -0.19013979,
             -0.17551816, -0.15421957, -0.12033667, 0.00522151, 0.05460073],
            [-0.19556192, -0.15759465, -0.0574587, 0.01521438, 0.06230766, 0.09036694,
             0.10248981, 0.09551204, 0.08906335, 0.07721034, 0.05254306, 0.01804762,
             -0.01980313, -0.08273582, -0.19623868, -0.58806378, -0.70225751]
        ])
    # plt.figure()
    # for prfl in profile_shapes:
    #     plt.plot(prfl, wind_data['altitude'])
    # plt.xlim([0, None])
    # plt.show()

    x_sol = np.empty((n_samples, 3))
    for i in range(n_samples):
        v = wind_speed[i:i+1, :]
        x0 = np.array([
            np.amax(v),
            0,
            0
        ])
        x_sol[i, :] = least_squares(fit_err, x0, args=(v, profile_shapes), bounds=((0, -np.inf, -np.inf), np.inf)).x
        print("{}/{}: {}".format(i, n_samples, x_sol[i, :]))
    np.save("x_sol_{}.npy".format(loc), x_sol)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_results(loc='mmij'):
    x_sol = np.load("x_sol_{}.npy".format(loc))

    ax1 = plt.figure().gca()
    sns.kdeplot(x=x_sol[:, 0], y=x_sol[:, 1], bw_adjust=.6, cmap=truncate_colormap(plt.get_cmap("Blues"), 0.2, 1.))
    ax2 = plt.figure().gca()
    sns.kdeplot(x=x_sol[:, 0], y=x_sol[:, 2], bw_adjust=.6, cmap=truncate_colormap(plt.get_cmap("Blues"), 0.2, 1.))

    altitudes = np.array([10., 20., 40., 60., 80., 100., 120., 140., 150., 160., 180., 200., 220., 250., 300., 500., 600.])
    if loc == 'mmc':
        rl = .1
        profile_shapes = np.array([
            log_law_wind_profile2(altitudes, rl, 1, 100, 0),
            [-0.23049645, -0.24630588, -0.28568619, -0.2984786, -0.30250081, -0.30061507,
             -0.29401967, -0.28247869, -0.27554446, -0.26734227, -0.25023463, -0.2305409,
             -0.20981732, -0.17730358, -0.12295292, 0.05224662, 0.10445487],
            [0.53796995, 0.4677825, 0.29697192, 0.15289426, 0.05002982, -0.03115965,
             -0.09539336, -0.14123821, -0.15740775, -0.16916436, -0.18807282, -0.19300634,
             -0.18839821, -0.16899669, -0.10964258, 0.20027731, 0.32476833]
        ])
    else:
        rl = .0002
        profile_shapes = np.array([
            log_law_wind_profile2(altitudes, rl, 1, 100, 0),
            [-0.3711465, -0.36080289, -0.33244151, -0.30885968, -0.28844681, -0.2697489,
             -0.25224161, -0.23609953, -0.22805606, -0.22041634, -0.20493771, -0.19013979,
             -0.17551816, -0.15421957, -0.12033667, 0.00522151, 0.05460073],
            [-0.19556192, -0.15759465, -0.0574587, 0.01521438, 0.06230766, 0.09036694,
             0.10248981, 0.09551204, 0.08906335, 0.07721034, 0.05254306, 0.01804762,
             -0.01980313, -0.08273582, -0.19623868, -0.58806378, -0.70225751]
        ])

    if loc == 'mmc':
        custom_cluster_profiles = np.array([
            [0.6803515, 0.71853229, 0.81305946, 0.85388069, 0.88120785, 0.90075635,
             0.9147152, 0.92376589, 0.92746348, 0.93017444, 0.93523973, 0.93869056,
             0.94166159, 0.94527828, 0.95007654, 0.9631492, 0.96909816],
            [0.50747916, 0.53380288, 0.59879481, 0.63002174, 0.65433224, 0.67529505,
             0.69420045, 0.71190688, 0.72080513, 0.72966774, 0.74756376, 0.76578488,
             0.7842986, 0.8123006, 0.85786185, 1.00233416, 1.04743931],
            [0.31997249, 0.3602069, 0.45884401, 0.5231481, 0.57224959, 0.6139435,
             0.65042426, 0.68224633, 0.69678184, 0.71047011, 0.73702799, 0.76101206,
             0.78314017, 0.81376994, 0.85699332, 0.96428156, 0.99027185],
            [0.31916794, 0.38611869, 0.55029902, 0.65804212, 0.73519261, 0.79609015,
             0.84480401, 0.88139352, 0.89594957, 0.90781971, 0.92918228, 0.94297737,
             0.95193377, 0.9588552, 0.95571666, 0.88390919, 0.84849594],
        ])
        for prfl in custom_cluster_profiles:
            x_cluster = np.zeros((5, 3))
            for i, v in enumerate([5, 10, 15, 20]):
                v_prfl = prfl*v
                x0 = np.array([
                    np.amax(v_prfl),
                    0,
                    0
                ])
                x_cluster[i+1, :] = least_squares(fit_err, x0, args=(v_prfl, profile_shapes), bounds=((0, -np.inf, -np.inf), np.inf)).x
            ax1.plot(x_cluster[:, 0], x_cluster[:, 1])
            ax2.plot(x_cluster[:, 0], x_cluster[:, 2])




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
    # eval_loc(loc)
    plot_results(loc)
    # plot_distr_pc12()
    # plot_results_pc1(loc)
    # plot_results_pc2(loc)
    plt.show()