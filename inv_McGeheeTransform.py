import numpy as np

def inv_McGeheeTransform(x2, x3, xdot, ydot, gamma, beta):
    r = np.abs(x2 + 1j * x3)**(1 / gamma)
    theta = np.angle(x2 + 1j * x3)
    v = r**(gamma * beta) * (xdot * np.cos(theta) + ydot * np.sin(theta))
    w = r**(gamma * beta) * (-xdot * np.sin(theta) + ydot * np.cos(theta))
    
    return r, theta, v, w

# Example usage:
# r, theta, v, w = inv_McGeheeTransform(x2, x3, xdot, ydot, gamma, beta)
