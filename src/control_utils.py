import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


def compute_discrete_system(A, B, dt):
    """
    Computes the corresponding discrete-time versions of A and B, where the current state is observed every dt units
    of time.
    :param A: The Jacobian of f with respect to the state vector, where dx/dt = f(x, u).
    :param B: The Jacobian of f with respect to the control input u (see the equation above).
    :param dt: The time gap in between subsequent observations of the state variable.
    :return: Two numpy arrays with identical shapes as A and B respectively.
    """
    # Compute discrete-time version of A
    A_tilde = expm(A*dt)

    # Compute discrete-time version of B using the Trapezoidal rule. See the "Uniform grid" section in
    # https://en.wikipedia.org/wiki/Trapezoidal_rule#Numerical_implementation
    B_tilde = np.zeros(B.shape)
    B_tilde += (expm(A*0) @ B)/2 + (expm(A*dt) @ B)/2

    num_sub_ints = 1000  # sub_ints = sub-intervals
    for k in range(1, num_sub_ints):
        tau = k / num_sub_ints * dt  # min = 1/num_parts * dt, max = (num_parts - 1)/num_parts * dt
        B_tilde += expm(A*tau) @ B

    B_tilde = dt/num_sub_ints * B_tilde

    return A_tilde, B_tilde
