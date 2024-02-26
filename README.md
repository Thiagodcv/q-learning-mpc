# Q-learning MPC
This repository contains an implementation of a data-driven MPC scheme. More specifically, we use Q-learning where the action-value function Q is approximated by the minimizer to a quadratic program. This quadratic program optimizes for a sequence of actions at each timestep, where the first of those actions will be applied to the system. This implementation is based off of [1] (section IV B.). Note that I intended to implement the version of this algorithm which includes a replay buffer, but never got around to it; doing so will likely improve the efficiency and stability of training. I found that [2] gives a clearer rundown of the problem setup. I apply this MPC scheme to the two-mass-spring system found in [3].

## Citations
[1] Gros, Sébastien, and Mario Zanon. "Data-driven economic NMPC using reinforcement learning." IEEE Transactions on Automatic Control 65.2 (2019): 636-648.

[2] Adhau, Saket, et al. "Fast Reinforcement Learning Based MPC based on NLP Sensitivities." IFAC-PapersOnLine 56.2 (2023): 11841-11846.

[3] Wie, Bong, and Dennis S. Bernstein. "Benchmark problems for robust control design." Journal of Guidance, Control, and Dynamics 15.5 (1992): 1057-1059.
