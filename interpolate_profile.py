import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import lagrange, interpolate

def lagrange_polynomial(tc, xc, t):
    n = len(tc)
    assert xc.shape[1] == n

    res = 0

    for j in range(n):
        p = 1
        for i in range(n):
            if i != j:
               p *= (t - tc[i]) / (tc[j] - tc[i])

        res += p * xc[:, j]

    return res


def lagrange_polynomial_2nd_der(tc, xc, t):
    n = len(tc)
    assert xc.shape[1] == n

    sum_j = 0
    for j in range(n):
        sum_i = 0
        for i in range(n):
            if i != j:
                sum_m = 0
                for m in range(n):
                    if m not in [i, j]:
                        p = 1
                        for sum_i in range(n):
                            if sum_i not in [i, j, m]:
                               p *= (t - tc[sum_i]) / (tc[j] - tc[sum_i])
                        sum_m += 1/(tc[j] - tc[m]) * p
                sum_i += 1/(tc[j] - tc[i]) * sum_m
        sum_j += xc[:, j] * sum_i
    return sum_j

v = np.array([0, 0.31997249, 0.3602069, 0.45884401, 0.5231481, 0.57224959, 0.6139435, 0.65042426, 0.68224633, 0.69678184,
     0.71047011, 0.73702799, 0.76101206, 0.78314017, 0.81376994, 0.85699332, 0.96428156, 0.99027185])
h = np.array([0, 10., 20., 40., 60., 80., 100., 120., 140., 150., 160., 180., 200., 220., 250., 300., 500., 600.])


def lagrange_interpolation():
    # def res(theta):
    #     zeroth = lagrange_polynomial(h, theta.reshape((1, -1)), h) - v
    #     second = 10**4*lagrange_polynomial_2nd_der(h, theta.reshape((1, -1)), h)
    #     return np.hstack((zeroth, second))

    sub = [1, 5, 11, 14, 17]

    step = 4
    end = -1
    h_sub = h[sub]
    h_fit, v_fit = zip(*[(hi, vi) for hi, vi in zip(h, v) if hi not in h_sub])
    v_sub = v[sub]
    # v_fit = v[1::step]

    def res(theta):
        zeroth = theta - v_sub
        poly = lagrange(h_sub, theta)
        fit = poly(h_fit) - v_fit
        # first = poly.deriv(1)(h_sub)
        # second = poly.deriv(2)(h_sub)
        return np.hstack((zeroth, fit))  #1000*second

    theta = least_squares(res, v_sub, verbose=2).x
    print(theta)
    print(res(theta))

    plt.plot(v, h, '*-')
    plt.plot(theta, h_sub, 's--', mfc='None')
    hf = np.linspace(10, 600.1, 100)
    # plt.plot(lagrange(h_sub, v[::i])(hf), hf, ':')
    plt.plot(lagrange(h_sub, theta)(hf), hf)
    # plt.plot(lagrange_polynomial(h, theta.reshape((1, -1)), hf), hf)
    plt.xlim([0, 1])
    plt.show()


def test_quadratic_interp():
    poly = np.poly1d([1, 4, 5])
    x = np.arange(3)
    y = poly(x)

    plt.plot(np.append(x, 3), np.append(y, 1.5), 's')
    f = interpolate.interp1d(np.append(x, 3), np.append(y, 1.5), 'quadratic')
    xf = np.linspace(0, 3, 100)
    plt.plot(xf, f(xf))
    plt.plot(xf, poly(xf))

if __name__ == '__main__':
    test_quadratic_interp()
    plt.show()