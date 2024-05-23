import numpy as np

def reverse_vector_field(_, Y, __, ___):
    m1 = 0.98784635699194478508644579051179  # mass 1
    m2 = 0.012153643008053502741483420379609  # mass 2
    m3 = 1.6540744733799014278142025561559e-15  # mass 3
    mu = m2 / (m1 + m2)  # mass ratio
    R3 = 1.18176e-7  # radius of m3
    C3_20 = -2.1151482476507870167381497594761  # oblateness stanford torus                
    c3 = ((m3**-(2/3)) * (R3**2) * C3_20) / 2  # radius scaling for m3
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
    ydot[0] = -(beta + 1) * Y[2] * Y[0]
    ydot[1] = -(Y[3] - Y[0])
    ydot[2] = -((beta * Y[2]**2 + Y[3]**2 - alpha * c) - Y[0]**(2 - 3 * gamma) -
                2 * A * (Y[0]**2) * np.cos(Y[1])**2 -
                2 * B * (Y[0]**2) * np.sin(Y[1])**2)
    ydot[3] = -((beta - 1) * Y[2] * Y[3] + 2 * (A - B) * (Y[0]**2) * np.sin(Y[1]) * np.cos(Y[1]))

    return ydot
