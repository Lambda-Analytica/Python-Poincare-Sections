import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Constants and function definitions
G = 1  # Gravitational constant (normalized)
m1 = 0.98784635699194478508644579051179  # mass 1
m2 = 0.012153643008053502741483420379609  # mass 2
m3 = 1.6540744733799014278142025561559e-15  # mass 3
mu = m2 / (m1 + m2)  # mass ratio
R1 = 0.016573881373569198521833456538843  # radius of m1
R2 = 0.0028069719042663893171507538681908  # radius of m2
R3 = 1.18176e-7  # radius of m3
C1_20 = -0.00134  # oblateness of m1
C2_20 = -0.00040  # oblateness of m2
C3_20 = -2.1151482476507870167381497594761  # oblateness stanford torus                
c1 = ((m3**(-2/3)) * (R1**2) * C1_20) / 2  # radius scaling for m1
c2 = ((m3**-(2/3)) * (R2**2) * C2_20) / 2  # radius scaling for m2
c3 = ((m3**-(2/3)) * (R3**2) * C3_20) / 2  # radius scaling for m3
C1 = (R1**2) * (C1_20) / 2
C2 = (R2**2) * (C2_20) / 2
C3 = (R3**2) * (C3_20) / 2
C12 = C1 + C2
C23 = C2 + C3
C13 = C1 + C3

# Functions
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

def inv_McGeheeTransform(x2, x3, xdot, ydot, gamma, beta):
    r = np.abs(x2 + 1j * x3)**(1 / gamma)
    theta = np.angle(x2 + 1j * x3)
    v = r**(gamma * beta) * (xdot * np.cos(theta) + ydot * np.sin(theta))
    w = r**(gamma * beta) * (-xdot * np.sin(theta) + ydot * np.cos(theta))
    return r, theta, v, w

def Hamiltonian_scalar(r, theta, v, w, A, B, gamma, beta, alpha, c):
    return (0.5 * r**(-2 * gamma * beta) * (v**2 + w**2) +
            r**(gamma * (1 - beta)) * np.sin(theta) * (v * np.cos(theta) - w * np.sin(theta)) -
            r**(gamma * (1 - beta)) * np.cos(theta) * (w * np.cos(theta) + v * np.sin(theta)) +
            A * r**(2 * gamma) * np.cos(theta)**2 + B * r**(2 * gamma) * np.sin(theta)**2 -
            r**(-gamma) - c * r**(gamma * alpha))

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

# Compute normalized distances and positions using the defined constants
omega = np.sqrt(1 - 3 * C12)
f_r13 = lambda r13: (1 / r13)**3 - 3 * C13 / r13**5 - omega**2
r13 = fsolve(f_r13, 1)[0]  # m1-m3 normalized distance
f_r23 = lambda r23: (1 / r23)**3 - 3 * C23 / r23**5 - omega**2
r23 = fsolve(f_r23, 1)[0]  # m2-m3 normalized distance

w1 = 1 + r13**2 - r23**2
beta = 3 / 2
gamma = 1 / (beta + 1)
c = -c3
alpha = 3

# Compute eigenvalues using the positions and constants
Delta = (mu * r13**3 + (1 - mu) * r23**3)**2 - mu * (1 - mu) * r13 * r23 * (
    -r13**4 - r23**4 + 2 * r13**2 + 2 * r23**2 + 2 * r13**2 * r23**2 - 1)
lambda1 = 0.5 * (2 - (2 * (1 - mu)) / r13**5 - (2 * mu) / r23**5 +
                 (3 * (1 - mu)) / r13**3 + (3 * mu) / r23**3 - 3 / (r13**3 * r23**3) * np.sqrt(Delta))  # eigen 1
lambda2 = 0.5 * (2 - (2 * (1 - mu)) / r13**5 - (2 * mu) / r23**5 +
                 (3 * (1 - mu)) / r13**3 + (3 * mu) / r23**3 + 3 / (r13**3 * r23**3) * np.sqrt(Delta))  # eigen 2

A = (1 - lambda2) / 2
B = (1 - lambda1) / 2

# Equilibria
E1 = [0.40358442529119564756712179587339, 0, 0, 0.0000000037549164666540642317192350962268]
E2 = [0.40358442529119564756712179587339, np.pi, 0, 0.0000000037549164666540642317192350962268]
E3 = [0.00000000076445101801200330861325273941074, np.pi/2, 0, 0.000000000000016887421337600002662639765488839]
E4 = [0.00000000076445101801200330861325273941074, 3*np.pi/2, 0, 0.000000000000016887421337600002662639765488839]

# Compute the energy at each libration point
CL1 = Hamiltonian_scalar(E1[0], E1[1], E1[2], E1[3], A, B, gamma, beta, alpha, c)
CL2 = Hamiltonian_scalar(E2[0], E2[1], E2[2], E2[3], A, B, gamma, beta, alpha, c)
CL3 = Hamiltonian_scalar(E3[0], E3[1], E3[2], E3[3], A, B, gamma, beta, alpha, c)
CL4 = Hamiltonian_scalar(E4[0], E4[1], E4[2], E4[3], A, B, gamma, beta, alpha, c)

# Compute positions of the three bodies in the coordinate system
x1 = -np.sqrt(m2**2 + w1 * m2 * m3 + r13**2 * m3**2)
y1 = 0
x2 = (-2 * m2**2 - 2 * r13**2 * m3**2 - 2 * w1 * m2 * m3 + 2 * m2 + w1 * m3) / (2 * np.sqrt(m2**2 + w1 * m2 * m3 + r13**2 * m3**2))
y2 = -0.5 * np.sqrt((m3**2 * (4 * r13**2 - w1**2)) / (m2**2 + w1 * m2 * m3 + r13**2 * m3**2))
x3 = (-2 * m2**2 - 2 * r13**2 * m3**2 - 2 * w1 * m2 * m3 + w1 * m2 + 2 * r13**2 * m3) / (2 * np.sqrt(m2**2 + w1 * m2 * m3 + r13**2 * m3**2))
y3 = 0.5 * np.sqrt((m2**2 * (4 * r13**2 - w1**2)) / (m2**2 + w1 * m2 * m3 + r13**2 * m3**2))

# Set energy level
C0 = 0.8 * CL1

# McGehee coordinate transformation
x = 0.35
y = 0
ydot = 0.2
xdot = np.sqrt(2 * (-0.5 * ydot**2 + ydot - A * x**2 + 1 / x - c3 / (x**3) + C0))

r0, theta0, v0, w0 = inv_McGeheeTransform(x2, x3, xdot, ydot, gamma, beta)

# Initialize parameters for the Poincaré map computation
k = 100
l = 1
iterates = 300

# Set left and right endpoints to be computed
x_begin = r0
x_end = r0 + 0.80

xdot_lower = 0.0
xdot_upper = 0.5

epsilon = 1e-6
timeStep = 10

# Define x-step value
if k > 1:
    xStep = (x_end - x_begin) / (k - 1)
else:
    xStep = 0

# Define xdot-step value, if l > 1 above
if l > 1:
    xdotStep = (xdot_upper - xdot_lower) / (l - 1)
else:
    xdotStep = 0

# Define the range for r0 variation
r0_min = r0 * 1.3
r0_max = r0 * 1.8

# Compute step size for r0 variation
r0_step = (r0_max - r0_min) / (k - 1)

# Initialize variables for the computation loop
x_nIterates = 0
totalIterates = 0
initial_n = 0
checkWhile = 0
interpTimes = 0
ReIntegrate = 0
checkTotal = 0
calledNewton = 0

PoincareMap = []

# Computation of Poincaré map
for m in range(l):
    for n in range(k):
        r0_current = r0_min + (n * r0_step)

        if m == 0:
            xdot0 = 0
        else:
            xdot0 = xdot_lower + ((m - 1) * (xdot_upper - xdot_lower) / (l - 1))

        initial_n = np.array([r0_current, theta0, v0, w0])

        x_nIterates = 0

        while x_nIterates <= iterates:
            checkWhile += 1

            tspan = [0, timeStep]
            sol = solve_ivp(vector_field, tspan, initial_n, args=(mu,), rtol=1e-13, atol=1e-13)
            trajectory_n = sol.y.T

            for i in range(1, len(trajectory_n)):
                if (np.sign(trajectory_n[i, 1]) != np.sign(trajectory_n[i - 1, 1])) and (trajectory_n[i, 3] > 0):
                    if np.abs(trajectory_n[i - 1, 1]) < epsilon:
                        intersection = trajectory_n[i - 1]
                    elif np.abs(trajectory_n[i, 1]) < epsilon:
                        intersection = trajectory_n[i]
                    else:
                        intersection = Trajectory_Crossing(sol.t[i - 1], trajectory_n[i - 1], sol.t[i], trajectory_n[i], epsilon, mu)
                        calledNewton += 1

                    x_nIterates += 1
                    totalIterates += 1
                    PoincareMap.append([intersection[0], intersection[3]])

            initial_n = trajectory_n[-1]

PoincareGrid = np.array(PoincareMap)

# Plot Poincaré section
plt.plot(PoincareGrid[:, 0], PoincareGrid[:, 1], 'b.')
plt.xlabel('r')
plt.ylabel('w')
plt.title('Poincaré Map')
plt.grid(True)
plt.show()

# Save the data
np.save('closeData2.npy', PoincareGrid)
# Load the data (example usage)
loaded_data = np.load('closeData2.npy')
