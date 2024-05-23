import numpy as np
from scipy.integrate import solve_ivp

def vector_field(t, Y, mu):
    m1 = 0.98784635699194478508644579051179
    m2 = 0.012153643008053502741483420379609
    m3 = 1.6540744733799014278142025561559e-15
    R3 = 1.18176e-7
    C3_20 = -2.1151482476507870167381497594761
    c3 = ((m3**-(2/3)) * (R3**2) * C3_20) / 2
    beta = 3/2
    gamma = 1 / (beta + 1)
    alpha = 2 * beta
    c = -c3
    r13 = 0.999999998424198
    r23 = 0.999999815955410
    Delta = ((mu * r13**3 + (1 - mu) * r23**3)**2 -
             mu * (1 - mu) * r13 * r23 * (-r13**4 - r23**4 + 2 * r13**2 + 2 * r23**2 + 2 * r13**2 * r23**2 - 1))

    sqrt_Delta = np.sqrt(Delta)
    lambda1 = 0.5 * (2 - (2 * (1 - mu)) / r13**5 - (2 * mu) / r23**5 +
                     (3 * (1 - mu)) / r13**3 + (3 * mu) / r23**3 - 3 / (r13**3 * r23**3) * sqrt_Delta)
    lambda2 = 0.5 * (2 - (2 * (1 - mu)) / r13**5 - (2 * mu) / r23**5 +
                     (3 * (1 - mu)) / r13**3 + (3 * mu) / r23**3 + 3 / (r13**3 * r23**3) * sqrt_Delta)

    A = (1 - lambda2) / 2
    B = (1 - lambda1) / 2

    ydot = np.zeros(4)
    ydot[0] = (beta + 1) * Y[2] * Y[0]
    ydot[1] = Y[3] - Y[0]
    ydot[2] = (beta * Y[2]**2 + Y[3]**2 - alpha * c - Y[0]**(2 - 3 * gamma) -
               2 * A * (Y[0]**2) * np.cos(Y[1])**2 -
               2 * B * (Y[0]**2) * np.sin(Y[1])**2)
    ydot[3] = (beta - 1) * Y[2] * Y[3] + 2 * (A - B) * (Y[0]**2) * np.sin(Y[1]) * np.cos(Y[1])

    return ydot

def reverse_vector_field(t, Y, mu):
    return -vector_field(t, Y, mu)

def Trajectory_Crossing(t0, initialPoint, t1, finalPoint, tolerance, mu):
    checkWhile = 0
    breakWhile = 16
    maxTime = 4
    newtonFailed = np.array([-mu, 0, 0, 0, 0, 0])

    t_n = t1 - t0

    if abs(finalPoint[1]) <= abs(initialPoint[1]):
        y_n = finalPoint[1]
        while abs(y_n) >= tolerance:
            if checkWhile < breakWhile and t_n < maxTime:
                tspan = [0, t_n]
                sol = solve_ivp(vector_field, tspan, finalPoint, args=(mu,), rtol=1e-6, atol=1e-13)
                f_n = sol.y[:, -1]
                fx_n = f_n[1]
                Dfx_n = f_n[3]

                t_n = t_n - fx_n / Dfx_n

                sol = solve_ivp(vector_field, tspan, finalPoint, args=(mu,), rtol=1e-6, atol=1e-13)
                f_n = sol.y[:, -1]
                y_n = f_n[1]
                crossing_n = f_n

                checkWhile += 1
            else:
                crossing_n = newtonFailed
                y_n = 0.1 * tolerance
    else:
        y_n = initialPoint[1]
        while abs(y_n) >= tolerance:
            if checkWhile < breakWhile and t_n < maxTime:
                tspan = [0, t_n]
                sol = solve_ivp(reverse_vector_field, tspan, initialPoint, args=(mu,), rtol=1e-6, atol=1e-13)
                f_n = sol.y[:, -1]
                fx_n = f_n[1]
                Dfx_n = -f_n[3]

                t_n = t_n - fx_n / Dfx_n

                sol = solve_ivp(reverse_vector_field, tspan, initialPoint, args=(mu,), rtol=1e-6, atol=1e-13)
                f_n = sol.y[:, -1]
                y_n = f_n[1]
                crossing_n = f_n

                checkWhile += 1
            else:
                crossing_n = newtonFailed
                y_n = 0.1 * tolerance

    return crossing_n

# Example usage:
# t0 = 0
# t1 = 10
# initialPoint = [1, 1, 0, 0]
# finalPoint = [1, -1, 0, 0]
# tolerance = 1e-5
# mu = 0.01215
# crossing_point = Trajectory_Crossing(t0, initialPoint, t1, finalPoint, tolerance, mu)
# print(crossing_point)
