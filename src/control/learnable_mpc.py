import logging
import numpy as np
import cvxpy as cp
import torch


class LearnableMPC:
    """
    An implementation of a model predictive controller with tunable parameters that can be optimized
    using Q-learning.

    References
    ----------
    [1] Gros, S., and Zanon, M. "Data-driven economic NMPC using reinforcement learning." IEEE Transactions
        on Automatic Control 65.2 (2019): 636-648.

    [2] Adhau, S., Reinhardt, D., Skogestad, S., and Gros, S. "Fast Reinforcement Learning Based MPC based on NLP
        Sensitivities." IFAC Preprint (2023).
    """

    def __init__(self, cost, dynamics, constraints, init_params, gamma,
                 weights, opt_horizon, dims, alpha, relax_input=False):
        """
        Parameters
        ----------
        cost : dict of functions
            The functions 'v_f' and 'l' which represent the terminal cost and stage cost, respectively.
        dynamics : function
            The dynamics function.
        constraints : dict of functions
            The functions 'g', 'h', and 'h_f' which represent constraints on the input signal, mixed constraints,
            and constraints on the state at the end of the prediction horizon.
        init_params : dict of arrays
            Parameterizes the above functions.
        gamma : float
            The discount factor in the objective function.
        weights : dict of arrays
            The weights 'w' and 'w_f' associated with relaxed constraints.
        opt_horizon : int
            The optimization horizon.
        dims : dict of ints
            Dimensions of state variable and input signal 'state_dim' and 'input_dim' respectively.
        alpha : dict
            Step sizes for parameter updates.
        relax_input : bool
            If true, relaxes control signal constraint g.
        """
        self.relax_input = relax_input
        self.true_l = cost['true_cost']
        self._v_f = cost['v_f']
        self._l = cost['l']

        self._f = dynamics

        self.g = constraints['g']
        self._h = constraints['h']
        self._h_f = constraints['h_f']

        self.w = weights['w']
        self.w_f = weights['w_f']
        if self.relax_input:
            self.w_g = weights['w_g']

        self.gamma = gamma
        self.N = opt_horizon
        self.alpha = alpha
        self.params = init_params

        self.state_dim = dims['state_dim']
        self.input_dim = dims['input_dim']
        self.h_dim = self.h(np.zeros(self.state_dim),
                            np.zeros(self.input_dim)).shape[0]
        self.h_f_dim = self.h_f(np.zeros(self.state_dim)).shape[0]
        self.g_dim = self.g(np.zeros(self.input_dim)).shape[0]

    def f(self, x, u):
        f_params = self.params['f']
        return self._f(x, u, f_params)

    def v_f(self, x):
        v_f_params = self.params['v_f']
        return self._v_f(x, v_f_params)

    def l(self, x, u):
        l_params = self.params['l']
        return self._l(x, u, l_params)

    def h(self, x, u):
        h_params = self.params['h']
        return self._h(x, u, h_params)

    def h_f(self, x):
        h_f_params = self.params['h_f']
        return self._h_f(x, h_f_params)

    def compute_value_function(self, state, action=None):
        """
        If action = None, will return state-value function evaluated at input state.
        If action is given, will return action-value function evaluated at input state and action.

        Parameters
        ----------
        state : (state_dim,) array
        action : (input_dim,) array, optional


        Return
        ------
        scalar : The value function evaluated at the optimal solution.
        dict : The primal variables in the optimal solution.
        dict : The dual variables in the optimal solution.
        """

        # Define variables to optimize over
        x = cp.Variable(shape=(self.state_dim, self.N + 1))
        u = cp.Variable(shape=(self.input_dim, self.N))
        sig = cp.Variable(shape=(self.h_dim, self.N))
        sig_N = cp.Variable(shape=self.h_f_dim)
        if self.relax_input:
            sig_u = cp.Variable(shape=(self.g_dim, self.N))

        # Define objective function
        obj = self.gamma**self.N * (self.v_f(x[:, self.N]) + self.w_f.T @ sig_N)
        for k in range(0, self.N):
            obj += self.gamma**k * (self.l(x[:, k], u[:, k]) + self.w.T @ sig[:, k])
            if self.relax_input:
                obj += self.gamma**k * self.w_g.T @ sig_u[:, k]

        obj = cp.Minimize(obj)

        # Define constraints
        chi_constr = [x[:, 0] == state]
        mu_constr = []
        nu_constr = []
        sig_constr = [0 <= sig_N]
        if self.relax_input:
            sig_u_constr = []

        for t in range(self.N):
            chi_constr += [x[:, t + 1] == self.f(x[:, t], u[:, t])]
            mu_constr += [self.h(x[:, t], u[:, t]) <= sig[:, t]]
            sig_constr += [0 <= sig[:, t]]
            if self.relax_input:
                nu_constr += [self.g(u[:, t]) <= sig_u[:, t]]
                sig_u_constr += [0 <= sig_u[:, t]]
            else:
                nu_constr += [self.g(u[:, t]) <= 0]

        mu_N_constr = [self.h_f(x[:, self.N]) <= sig_N]

        if action is not None:
            zeta_constr = [u[:, 0] == action]
            constr = chi_constr + mu_constr + mu_N_constr + nu_constr + sig_constr + zeta_constr
        else:
            constr = chi_constr + mu_constr + mu_N_constr + nu_constr + sig_constr

        if self.relax_input:
            constr += sig_u_constr

        # Solve optimization (21) or (23) of [1]
        prob = cp.Problem(obj, constr)
        prob.solve(verbose=False)

        # Ensure optimization is correctly solved
        if prob.status != 'optimal':
            print("Uh oh.")

        # Retrieve dual values
        chi_duals = np.zeros(shape=(self.state_dim, self.N+1))
        mu_duals = np.zeros(shape=(self.h_dim, self.N))
        mu_N_dual = mu_N_constr[0].dual_value
        nu_duals = np.zeros(shape=(self.g_dim, self.N))
        if action is not None:
            zeta_dual = zeta_constr[0].dual_value
        else:
            zeta_dual = None

        chi_duals[:, 0] = chi_constr[0].dual_value
        for t in range(self.N):
            chi_duals[:, t+1] = chi_constr[t+1].dual_value
            mu_duals[:, t] = mu_constr[t].dual_value
            nu_duals[:, t] = nu_constr[t].dual_value

        # Package and return solutions
        min_v = obj.value
        primal = {'x': np.array(x.value),
                  'u': np.array(u.value),
                  'sig': np.array(sig.value)}

        if self.relax_input:
            primal['sig_u'] = np.array(sig_u.value)

        dual = {'chi': chi_duals,
                'mu': mu_duals,
                'mu_N': mu_N_dual,
                'nu': nu_duals,
                'zeta': zeta_dual}

        return min_v, primal, dual

    def compute_grad_action_value_function(self, state, action):
        """
        Computes the Lagrangian of problem (23) in [1], and subsequently uses autograd
        to find gradient (29) of the action-value function.

        Parameters
        ----------
        state: (state_dim,) array
        action: (input_dim,) array

        Return
        ------
        float: the action-value function evaluated at the optimal solution.
        float: the Lagrangian of problem (23) evaluated at the optimal solution.
        """
        # Computing y* in (29) of [1]
        q, primal, dual = self.compute_value_function(state, action)
        x, u = primal['x'], primal['u']
        chi, mu, mu_N, nu, zeta = dual.values()

        # Convert numpy arrays into torch.tensors
        state = torch.from_numpy(state).float()
        action = torch.from_numpy(action).float()

        x = torch.from_numpy(x).float()
        u = torch.from_numpy(u).float()
        chi = torch.from_numpy(chi).float()
        mu = torch.from_numpy(mu).float()
        mu_N = torch.from_numpy(mu_N).float()
        nu = torch.from_numpy(nu).float()
        zeta = torch.from_numpy(zeta).float()

        # Compute lagrangian (28) of [1]
        lagrangian = self.gamma**self.N * self.v_f(x[:, self.N]) + torch.matmul(chi[:, 0], (x[:, 0] - state)) + \
                     torch.matmul(mu_N, self.h_f(x[:, self.N])) + torch.matmul(zeta, (u[:, 0] - action))

        for k in range(self.N):
            lagrangian += torch.matmul(chi[:, k+1], (self.f(x[:, k], u[:, k]) - x[:, k+1])) + \
                          torch.matmul(nu[:, k], self.g(u[:, k])) + self.gamma**k * self.l(x[:, k], u[:, k]) + \
                          torch.matmul(mu[:, k], self.h(x[:, k], u[:, k]))

        # Compute derivative of lagrangian w.r.t model parameters
        lagrangian.backward()

        return q, lagrangian

    def temp_diff_step(self, state, action, next_state):
        """
        Updates the model parameters using the temporal difference update (19) in [2], and subsequently
        zeros out the gradients for each model parameter
        (see https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch).

        Parameters
        ----------
        state : (state_dim,) array
        action : (input_dim,) array
        next_state : (state_dim,) array
        """
        # Compute action-value at current (x_k, u_k), set gradients w.r.t model parameter
        q_curr, _ = self.compute_grad_action_value_function(state, action)
        v_next, _, _ = self.compute_value_function(next_state)

        temp_diff = self.true_l(state, action) + self.gamma * v_next - q_curr

        # Iterate through all model parameters
        for func, func_params in self.params.items():
            for param in func_params.values():
                # Only update if tensor.requires_grad == True
                if param.requires_grad:
                    # NOTE: If the line below is run without torch.no_grad(), the model parameter will cease to be a
                    # leaf, which means that its .grad attribute won't be populated during future autograd.backward()
                    # calls.
                    with torch.no_grad():
                        # Update model parameters. Clipping is used to prevent exploding weight values.
                        param += torch.clip(self.alpha[func] * temp_diff * param.grad.data, min=-1, max=1)
                    # Zero-out the gradient
                    param.grad.data.zero_()

        # print('A: \n')
        # print(self.params['f']['A'].detach().numpy())
        # print('B: \n')
        # print(self.params['f']['B'].detach().numpy())
        # print('V_chol: \n')
        # print(self.params['v_f']['V_chol'].detach().numpy())
        # print('C_chol: \n')
        # print(self.params['l']['C_chol'].detach().numpy())

    def train(self, env, num_episodes, episode_len, epsilon, eval_num=5):
        """
        Learn an MPC controller using Q-learning as outlined in section IV. B. of [1].

        Parameters
        ----------
        env: gymnasium environment
        num_episodes: int
            Number of episodes to sample.
        episode_len: int
            Number of time steps in each episode.
        epsilon: float
            Number in (0, 1) representing the probability of choosing a random action
            (and not following the optimal policy).
        eval_num: int
            Number of episodes between the evaluation of the current policy (without greedy exploration).
        """
        for ep in range(num_episodes):
            obs, info = env.reset()
            for t in range(episode_len):
                # With probability 1-epsilon choose a greedy strategy
                if np.random.uniform(low=0, high=1, size=1) > epsilon:
                    _, primal, _ = self.compute_value_function(state=obs)
                    action = primal['u'][:, 0]
                else:
                    action = np.random.normal(loc=0, scale=0.6, size=1)

                next_obs, reward, terminated, truncated, _ = env.step(action)

                # Update parameters
                self.temp_diff_step(state=obs, action=action, next_state=next_obs)
                obs = next_obs

                if terminated or truncated:
                    break

            env.close()

            # Evaluate greedy policy after every 'eval_num' number of episodes
            if ep % eval_num == 0:
                disc_return = self.simulate_env(env=env, episode_len=episode_len)
                print("Discounted sum of rewards of current policy at timestep {}: {}".format(ep, disc_return))
                self.print_model_parameters()

            print("------------------------Episode {} Terminated------------------------".format(ep))

    def print_model_parameters(self):
        for func, func_params in self.params.items():
            print(func + ':')
            for param_name, param in func_params.items():
                print('    ' + param_name + ':')
                print(param.detach().numpy())
                print('\n')

    def simulate_env(self, env, episode_len, seed=None):
        """
        Simulates the controller in a gymnasium environment.

        Parameters
        ----------
        env: gymnasium environment
        episode_len: int
            The time horizon for which environment is run for.
        seed: int
            Seed for the environment's initial state variable.

        Return
        ------
        float: The realized sum of discounted rewards.
        """
        if seed is None:
            obs, info = env.reset()
        else:
            obs, info = env.reset(seed=seed)
        rewards = []

        for t in range(episode_len):
            _, primal, _ = self.compute_value_function(state=obs)
            action = primal['u'][:, 0]
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break
        env.close()
        disc_return = np.inner([self.gamma ** k for k in range(episode_len)], rewards)
        return disc_return
