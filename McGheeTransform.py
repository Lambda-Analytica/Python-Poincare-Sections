import numpy as np

def McGeheeTransform(r, theta, v, w, gamma, beta):
    x = (r**gamma) * np.cos(theta)
    y = (r**gamma) * np.sin(theta)
    xdot = (r**(-gamma * beta)) * (v * np.cos(theta) - w * np.sin(theta))
    ydot = (r**(-gamma * beta)) * (v * np.sin(theta) + w * np.cos(theta))
    return x, y, xdot, ydot

# Example usage:
# r = 1.0
# theta = np.pi / 4
# v = 0.5
# w = 0.3
# gamma = 1.0
# beta = 0.5
# x, y, xdot, ydot = McGeheeTransform(r, theta, v, w, gamma, beta)
# print(f"x: {x}, y: {y}, xdot: {xdot}, ydot: {ydot}")
