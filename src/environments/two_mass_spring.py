import gymnasium as gym
import numpy as np
from typing import Optional
from control_utils import compute_discrete_system


class TwoMassSpring(gym.Env):
    """
    An implementation of the two-mass-spring system introduced in

    [1] Wie, Bong, and Dennis S. Bernstein. "Benchmark problems for robust control design." Journal of Guidance,
        Control, and Dynamics 15.5 (1992): 1057-1059.
    """

    def __init__(self, params, init_state, render_mode: Optional[str] = None, episode_len=200):
        """
        Parameters
        ----------
        params: dict
            Contains the follow key-value pairs:
                k: float
                    The spring constant.
                m1: float
                    The mass of the left trolley.
                m2: float
                    The mass of the right trolley.
                targ: float
                    The coordinate where we want the right trolley to be positioned at using control mechanisms.
        init_state: array
            The initial state the environment starts in.
        render_mode: None
            Not implemented.
        episode_len: int
            Number of time steps in environment.
        """
        # The parameters of the environment
        self.k = params['k']
        self.m1 = params['m1']
        self.m2 = params['m2']
        self.targ = params['targ']

        self.init_state = init_state
        self.state = None
        self.curr_t = None
        self.episode_len = episode_len

        self.A_tilde = np.array([[0, 0, 1, 0],
                                 [0, 0, 0, 1],
                                 [-self.k/self.m1, self.k/self.m1, 0, 0],
                                 [self.k/self.m2, -self.k/self.m2, 0, 0]])
        self.B_tilde = np.array([[0],
                                 [0],
                                 [1/self.m1],
                                 [0]])
        self.state_dim = self.A_tilde.shape[1]
        self.input_dim = self.B_tilde.shape[1]

        self.dt = 1  # 0.05
        self.A, self.B = compute_discrete_system(self.A_tilde, self.B_tilde, self.dt)

    def running_cost(self, x, u):
        """
        Cost punishes distance of second trolley from target, and energy expenditure at each timestep.
        Velocity of the trolleys not punished.
        """
        cost = (x[1] - self.targ)**2 + u**2
        return cost[0]

    def terminal_cost(self, x):
        return (x[1] - self.targ)**2

    def step(self, u):
        self.state = self.A @ self.state + self.B @ u
        self.curr_t += 1

        if self.curr_t == self.episode_len:
            return self._get_obs(), self.terminal_cost(self.state), True, False, {}
        else:
            return self._get_obs(), self.running_cost(self.state, u), False, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.curr_t = 0
        self.state = self.init_state

        return self._get_obs(), {}

    def _get_obs(self):
        return self.state

    def render(self):
        pass

    def close(self):
        pass
