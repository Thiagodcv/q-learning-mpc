import torch
import numpy as np
import cvxpy as cp
from control.learnable_mpc import LearnableMPC
from environments.two_mass_spring import TwoMassSpring
from control_utils import compute_discrete_system


def mpc_two_mass_spring():
    # True values used in environment
    k_true = 1
    m1_true = 1
    m2_true = 1

    # Inaccurate parameter estimates used in MPC
    k = 1.5
    m1 = 0.8
    m2 = 1.2
    # k = k_true
    # m1 = m1_true
    # m2 = m2_true

    # Parameters for dynamics function
    A_tilde = np.array([[0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [-k/m1, k/m1, 0, 0],
                        [k/m2, -k/m2, 0, 0]])
    B_tilde = np.array([[0],
                        [0],
                        [1/m1],
                        [0]])
    state_dim = A_tilde.shape[1]
    input_dim = B_tilde.shape[1]

    dt = 1  # 0.05
    A, B = compute_discrete_system(A_tilde, B_tilde, dt)
    A = torch.tensor(A, requires_grad=False).float()
    B = torch.tensor(B, requires_grad=False).float()

    targ = 10

    # Log-Cholesky decomposition parameters for terminal cost function. Must be upper-triangular (diagonal will be exp)
    V_chol = torch.tensor([[-1e5, 0., 0., 0.],
                           [0., 0., 0., 0.],
                           [0., 0., -1e5, 0.],
                           [0., 0., 0., -1e5]], requires_grad=True)

    v_vec = torch.tensor([0., targ, 0., 0.], requires_grad=True)

    C_chol = torch.tensor([[-1e5, 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., -1e5, 0., 0.],
                           [0., 0., 0., -1e5, 0.],
                           [0., 0., 0., 0., 0.]], requires_grad=True)

    c_vec = torch.tensor([0., targ, 0., 0., 0.], requires_grad=True)

    # Parameters for mixed constraints (for now only constrain x)
    h_mat = torch.tensor([0., 0., 0., 0.], requires_grad=False)
    h_vec = torch.tensor([0.], requires_grad=False)

    # Parameters for terminal state constraints (currently not active)
    h_f_mat = torch.tensor([0., 0., 0., 0.], requires_grad=False)
    h_f_vec = torch.tensor([0.], requires_grad=False)

    # Constraint on absolute value of control signal
    u_const = 5

    def f(x, u, params):
        """
        Note that A and B are torch tensors.
        """
        A = params['A']
        B = params['B']
        if isinstance(x, torch.Tensor):
            return torch.matmul(A, x) + torch.matmul(B, u)
        else:  # isinstance(x, cp.Expression):
            return A.detach().numpy() @ x + B.detach().numpy() @ u

    def v_f(x, params):
        """
        Terminal cost.
        """
        # torch.triu ensures only upper-triangular part is updated.
        V_chol = torch.triu(params['V_chol'])
        v_vec = params['v_vec']
        n_x = V_chol.shape[0]

        # Let V be equal to V_chol but with all diagonal elements exponentiated.
        V = torch.diag(torch.exp(torch.diag(V_chol))) + (1. - torch.eye(n_x)) * V_chol

        U = torch.matmul(torch.transpose(V, 0, 1), V) + 1e-2 * torch.eye(n_x)
        u_vec = torch.matmul(v_vec, V)
        u_scalar = torch.matmul(v_vec, v_vec)

        if isinstance(x, torch.Tensor):
            return torch.matmul(torch.matmul(x, U), x) - 2*torch.matmul(u_vec, x) + u_scalar
        else:
            U_np = U.detach().numpy()
            u_vec_np = u_vec.detach().numpy()
            u_scalar_np = u_scalar.detach().numpy()
            return cp.quad_form(x, U_np) - 2*u_vec_np.T @ x + u_scalar_np

    def l(x, u, params):
        """
        Learnable stage cost.
        """
        # torch.triu ensures only upper-triangular part is updated.
        C_chol = torch.triu(params['C_chol'])
        c_vec = params['c_vec']
        n_z = C_chol.shape[0]

        C = torch.diag(torch.exp(torch.diag(C_chol))) + (1. - torch.eye(n_z)) * C_chol

        R = torch.matmul(torch.transpose(C, 0, 1), C) + 1e-2 * torch.eye(n_z)
        r_vec = torch.matmul(c_vec, C)
        r_scalar = torch.matmul(c_vec, c_vec)

        if isinstance(x, torch.Tensor):
            z = torch.hstack((x, u))
            return torch.matmul(torch.matmul(z, R), z) - 2*torch.matmul(r_vec, z) + r_scalar
        else:
            R_np = R.detach().numpy()
            r_vec_np = r_vec.detach().numpy()
            r_scalar_np = r_scalar.detach().numpy()
            z = cp.hstack([x, u])
            return cp.quad_form(z, R_np) - 2*r_vec_np.T @ z + r_scalar_np

    def g(u):
        # Control signal constraint, so g(u) <= 0.
        if isinstance(u, torch.Tensor):
            id = torch.eye(input_dim)
            return torch.matmul(torch.vstack((id, -id)), u) - u_const * torch.ones(2 * input_dim)
        else:
            id = np.identity(input_dim)
            return np.vstack((id, -id)) @ u - u_const * np.ones(2 * input_dim)

    def h(x, u, params):
        """
        Mixed constraints, where h(x,u) <= 0.
        """
        h_mat = params['h_mat']
        h_vec = params['h_vec']

        if isinstance(x, torch.Tensor):
            return torch.matmul(h_mat, x) - h_vec
        else:
            h_mat_np = h_mat.detach().numpy()
            h_vec_np = h_vec.detach().numpy()
            return h_mat_np @ x - h_vec_np

    def h_f(x, params):
        """
        Terminal state constraint, where h_f(x) <= 0.
        """
        h_f_mat = params['h_f_mat']
        h_f_vec = params['h_f_vec']

        if isinstance(x, torch.Tensor):
            # return torch.zeros(size=(1,))
            return torch.matmul(h_f_mat, x) - h_f_vec
        else:
            # return np.zeros(shape=(1,))
            h_f_mat_np = h_f_mat.detach().numpy()
            h_f_vec_np = h_f_vec.detach().numpy()
            return h_f_mat_np @ x - h_f_vec_np

    def true_cost(x, u):
        """
        The true stage cost associated with the MDP (not tunable).
        """
        cost = (x[1] - targ)**2 + u**2
        return cost[0]

    cost = {'v_f': v_f, 'l': l, 'true_cost': true_cost}
    dynamics = f
    constraints = {'g': g, 'h': h, 'h_f': h_f}
    init_params = {'f': {'A': A, 'B': B},
                   'v_f': {'V_chol': V_chol, 'v_vec': v_vec},
                   'l': {'C_chol': C_chol, 'c_vec': c_vec},
                   'h': {'h_mat': h_mat, 'h_vec': h_vec},
                   'h_f': {'h_f_mat': h_f_mat, 'h_f_vec': h_f_vec}}
    gamma = 0.99
    weights = {'w': 100 * np.ones(1), 'w_f': 100 * np.ones(1), 'w_g': 100 * np.ones(2)}
    opt_horizon = 10
    dims = {'state_dim': state_dim, 'input_dim': input_dim}
    alpha = {'f': 1e-9,
             'v_f': 1e-6,
             'l': 1e-6,
             'h': 1e-6,
             'h_f': 1e-6}

    mpc = LearnableMPC(cost=cost,
                       dynamics=dynamics,
                       constraints=constraints,
                       init_params=init_params,
                       gamma=gamma,
                       weights=weights,
                       opt_horizon=opt_horizon,
                       dims=dims,
                       alpha=alpha,
                       relax_input=False)

    # Define the environment
    init_state = np.array([0., 2., 0., 0.])
    env_params = {'k': k_true, 'm1': m1_true, 'm2': m2_true, 'targ': targ}
    env = TwoMassSpring(params=env_params, init_state=init_state, episode_len=30)

    # disc_ret = mpc.simulate_env(env=env, episode_len=30)
    # print('Discounted return:', disc_ret)

    mpc.train(env=env, num_episodes=200, episode_len=30, epsilon=0.2)


if __name__ == '__main__':
    mpc_two_mass_spring()
