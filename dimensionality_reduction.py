import numpy as np


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


def log_law_wind_profile3(z, z_0, v_fric, ol):
    beta = 6
    gamma = 19.3
    kappa = .41

    if ol < 0:
        def psi(z, ol):
            x = (1 - gamma * z / ol) ** .25
            psi = 2 * np.log((1 + x) / 2) + np.log((1 + x ** 2) / 2) - 2 * np.arctan(x) + np.pi / 2
            return psi

        v = v_fric/kappa * (np.log(z / z_0) - psi(z, ol))
    elif ol == 0:
        v = v_fric/kappa * np.log(z / z_0)
    else:
        def psi(z, ol):
            psi = - beta * z / ol
            return psi

        v = v_fric/kappa * (np.log(z / z_0) - psi(z, ol))
    return v


def fit_err(x, wind_speed, profile_shapes):
    v_err = wind_speed - np.dot(x.reshape((1, -1)), profile_shapes)
    v_err = v_err.reshape(-1)
    return v_err