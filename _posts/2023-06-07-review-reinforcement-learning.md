---
layout: post
title: "Review: Reinforcement Learning"
categories: machine_learning
tags: [machine_learning, deep_learning]
math: true

---

## Tabular Value Based Methods
### Backup
- Monte-Carlo backup: zero bias, high variance.

$$
\begin{align*}
V(S_t) \leftarrow V(S_t) + \alpha(G_t - V(S_t))
\end{align*}
$$

- Temporal difference backup: high bias, low variance.

$$
\begin{align*}
V(S_t) &\leftarrow V(S_t) + \alpha(R_{t + 1} + \gamma V(S_{t + 1}) - V(S_t))\\
V(S_{t}) &\leftarrow \alpha [R_{t + 1} + \gamma V(S_{t + 1})] + (1 - \alpha) V(S_t)
\end{align*}
$$

- Dynamic programming backup.

$$
\begin{align*}
V(S_t) \leftarrow E_{\pi}[R_{t + 1} + \gamma V(S_{t + 1})]
\end{align*}
$$

### On / Off Policy
- On-behavior-policy learning up: SARSA.

$$
\begin{align*}
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[r_{t + 1} + \gamma Q(s_{t + 1}, a_{t + 1})- Q(s_t, a_t)]
\end{align*}
$$

- Off-behavior-policy learning up: Q-learning.

$$
\begin{align*}
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[r_{t + 1} + \gamma \max_a Q(s_{t + 1}, a)- Q(s_t, a_t)]
\end{align*}
$$

### MDP Objective
$$
\begin{align*}
V^{\pi}(s) &= E_{\tau_t\sim p(\tau_t)}\left[\sum_{i = 0}^{\infty} \gamma^i\cdot r_{t + i}| s_t = s \right]\\
Q^{\pi}(s, a) &= E_{\tau_t\sim p(\tau_t)}\left[\sum_{i = 0}^{\infty} \gamma^i\cdot r_{t + i}| s_t = s, a_t = a \right]\\
\pi^* &= \text{argmax}_{\pi} V^{\pi}(s) = \text{argmax}_{a,\pi} Q^{\pi} (s, a)\\
a^* &= \text{argmax}_{a\in A} Q^*(s, a)
\end{align*}
$$

Bellman equation:

$$
\begin{align*}
V^{\pi}(s) = \sum_{a\in A} \pi(a|s)\left[\sum_{s'\in S} T_a(s, s')\left[R_a(s, s') + \gamma\cdot V^{\pi}(s') \right] \right]
\end{align*}
$$

## Deep Value Based Methods
**Dead Triad**\
Value function approximation is unstable:
1. Function approximation (Deep).
2. Off-policy learning (Q-learning).
3. Bootstrapping (TD).

### Three Problems
1. Coverage.\
    Convergence proof of value iteration, Q-learning and SARSA depends on covering the entire state space, in the end. Not even close, in high dimensional problems.
2. Correlation.
    - Supervised: Database examples are uncorrelated. Stable learning.
    - Deep: Actions determine next state that will be learned from.
3. Convergence.
    - Supervised: Minimization loss-target $$y$$ is fixed.
    - Reinforcement: Convergence loss-target $$Q_{t−1}$$ is moving.
    - Converging on a moving target is hard.

#### Solutions in DQN
1. Coverage: High exploration.
2. Correlation: Replay buffer --- de-correlation of examples.
    - Store all experience in buffer.
    - Sample from the history buffer.
    - Choose action epsilon greedy.
    - Add some SL to RL.
3. Convergence: Low $$\alpha$$, infrequent weight updates --- slow learning.
    - Introduce a separate **target network** for the convergence targets.
    - Every $$c$$ updates, clone the network to a target network.
    - Add delay between updates of the network. Use updates in other states.
    - Reduce oscillations or divergence of the policy.

### Methods
- DQN: Baseline.
- Double DQN: De-overestimate values.
- Dueling DDQN: Advantage function $$A(s, a) = Q(s, a) - V(s)$$ to standardize action values.
- Prioritized experience: Sort replay buffer history.
- A3C: Parallel actor critic.
- Distributional DQN: Probability distribution.
- Noisy DQN: Add parametric noise to increase exploration.

|Name|Principle|Applicability|Effectiveness|
|:---:|:---:|:---:|:---:|
|DQN|replay buffer|Atari|stable Q learning|
|Double DQN|de-overestimate values|DQN|convergence|
|Prioritized experience|decorrelation|replay buffer|convergence|
|Distributional|probability distribution|stable gradients|generalization|
|Random noise|parametric noise|stable gradients|more exploration|

## Policy Based Reinforcement Learning
### Tabular Versus Function Approximation
**Table**
- Exact.
- Easy to design.
- Curse of dimensionality.
- No generalization.

**Function approximation**
- Generalization.
- Lower memory requirement (scales to high-dim).
- Approximation errors.

Type of space / Dimensionality

||Discrete|Continuous|
|:---:|:---:|:---:|
|Low dimensional|Table / FA|Discretization / FA|
|High dimensional|FA|FA|

### Actor-critic
The value function supports the update of the policy:
1. Through bootstrapping: Lower variance in cumulative reward estimate.
2. Through baseline subtraction: Lower variance in gradient estimate.
3. As a direct target for optimization: Deterministic policy gradient.
    - Train critic in the same way.
    - Train actor by pushing gradient through the actions.

    $$
    \begin{align*}
    J(\theta) &= E_{s\sim D}\left[\sum_{t = 0}^n Q_\phi (s, \pi_{\theta}(s)) \right]\\
    \bigtriangledown_\theta J(\theta) &= \sum_{t = 0}^n\bigtriangledown_a Q_\phi(s, a)\cdot \bigtriangledown_\theta \pi_\theta(s)
    \end{align*}
    $$

### Policy Gradient
$$
\begin{align*}
\bigtriangledown_\theta J(\theta) &= \bigtriangledown_\theta E_{h_0\sim p_\theta (h_0)}[R(h_0)]\\
&= E_{h_0\sim p_\theta(h_0)}[R(h_0)\cdot \bigtriangledown_\theta \log p_\theta(h_0)]\\
&= E_{h_0\sim p_\theta(h_0)}\left[R(h_0)\cdot \sum_{t=0}^n \bigtriangledown_\theta \log \pi_{\theta}(a_t|s_t) \right]
\end{align*}
$$

**Policy gradient theorem (REINFORCE)**\
Make sense to move $$R(h_0)$$ inside the sum, since return of action only depends on future rewards.

$$
\begin{align*}
\bigtriangledown_\theta E_{h_0\sim p_{\theta}(h_0)}[R(h_0)] = E_{h_0\sim p_\theta(h_0)}\left[\sum_{t = 0}^n R(h_t)\bigtriangledown_\theta \log \pi_\theta (a_t|s_t) \right]
\end{align*}
$$

**General policy gradient formulation**

$$
\begin{align*}
\bigtriangledown_\theta J(\theta) = E_{h_0\sim p_\theta(h_0)}\left[\sum_{t=0}^n \Psi_t \bigtriangledown_\theta \log \pi_\theta (a_t|s_t) \right]
\end{align*}
$$

- Monte Carlo target
    $$
    \begin{align*}
    \Psi_t = \hat{Q}_{MC}(s_t, a_t) = \sum_{i = t}^{\infty} \gamma^i\cdot r_i
    \end{align*}
    $$
- Bootstrapping (n-step target)
    $$
    \begin{align*}
    \Psi_t = \hat{Q}_{n}(s_t, a_t) = \sum_{i = t}^{n - 1} \gamma^i\cdot r_i + \gamma^n V_\theta(s_n)
    \end{align*}
    $$
- Baseline subtraction
    $$
    \begin{align*}
    \Psi_t = \hat{Q}_{n}(s_t, a_t) = \sum_{i = t}^{\infty} \gamma^i\cdot r_i - V_\theta(s_t)
    \end{align*}
    $$
- Baseline subtraction + bootstrapping
    $$
    \begin{align*}
    \Psi_t = \hat{Q}_{n}(s_t, a_t) = \sum_{i = t}^{n - 1} \gamma^i\cdot r_i + \gamma^n V_\theta(s_n) - V_\theta(s_t)
    \end{align*}
    $$
- Q-value approximation
    $$
    \begin{align*}
    \Psi_t = Q_\theta(s_t, a_t)
    \end{align*}
    $$

### Gradient-free Policy Search
- Evolutionary strategies
- Cross-entropy method
    1. Start with the normal distribution $$N(\mu, \sigma^2)$$.
    2. Evaluate some parameters from this distribution and select the best.
    3. Compute the mean and standard deviation of the best. Add some noise and go to step 1.
- Simulated annealing

### Policy Based Agents

|Name|Approach|
|:---:|:---:|
|REINFORCE|Policy-gradient optimization|
|A3C|Distributed actor critic|
|DDPG|Derivative of continuous action function|
|TRPO|Dynamically sized step size|
|PPO|Improved TRPO, first order|
|SAC|Variance-based actor critic for robustness|

Enhancements to reduce high variance:
- Actor critic introduces within-episode value-based critics based on temporal difference value bootstrapping.
- Baseline subtraction introduces an advantage function to lower variance.
- Trust regions reduce large policy parameter changes.
- Exploration is crucial to get out of local minima. And for more robust result, high entropy action distributions are often used.

## Model Based Reinforcement Learning
**Learning**
- Agent changing state in the environment.
- Irreversible state change.
- Forward path.

**Planning**
- Agent changing own local state.
- Reversible local state change.
- Backtracking Tree.

**Classic approach: Dyna**
- Dyna’s tabular imagination.
- Environment samples are used in a **hybrid** model-free / model-based manner, to train the transition model, use planning to improve the policy, while also training the policy function directly.
- This hybrid model-based planning is called imagination because looking ahead with the agent’s own dynamics model resembles imagining environment samples outside the real environment inside the “mind” of the agent. 
- In this approach the imagined samples augment the real (environment) samples at no sample cost.

### Imperfect Models
#### Improving Model Learning
- Modeling uncertainty
    - Knowing uncertainty allows better planning. Smart methods from statistics for better sampling.
    - Do planning sampling from distribution, plan with locally-linear search or with stochastic trajectory optimizer.
    - Do not scale to high dimensional problems.
    - Gaussian processes: Works but computationally expensive. e.g. PILCO, GPS, SVG.
    - Ensembles. e.g. PETS.
- Latent models
    - Compress observation space into (smaller) latent space by modeling on value prediction. And plan in small latent space. 
    - Reduce observation space to essence.
    - Examples: PlaNet, Dreamer, VPN.

#### Improving Planning
- Trajectory rollouts and model-predictive control (MPC)
    - Short trajectory rollouts: Reduce lookahead depth. Split rollouts in near future (planned) and far-future (model-free). $$\Rightarrow$$ Model-based value expansion (MVE).
    -  Model-predictive control 
        - Decision-time planning.
        - Highly non-linear function are often locally linear.
        - MPC: optimize model over limited time, and re-learn. $$\Rightarrow$$ PETS.
- End-to-end learning and planning
    - Learn differentiable planning.
        - “Impedance mismatch”.
        - Model is learned 1-step.
        - Planning looks-ahead n-step.
        - Integrated end-to-end learning of model and planning matches model and planner.
    - Example: Value iteration network (VIN), VProp, TreeQN, Predictron, MuZero, I2A, World Model.

#### Summary of Models

|||Learning||
|:---:|:---:|:---:|:---:|
|||Uncertainty ensembles|Latent models|
|**Planning**|Trajectory / MPC| **PILCO**, GPS, SVG, Local, **PETS**, MVE, Meta|**Deamer**, Plan2Explore, L3P, **VPN**, SimPle, Dreamer-v2|
||End-to-end|**VIN**, **Vprop**, Network-Planning|TreeQN, **I2A**, Predictron, World Model, **MuZero**|

- PILCO: Uncertainty / trajectory, Gaussian processes. Computationally expensive.
- PETS: Ensemble, MPC.
- VPN: Latent / trajectory.
- Dreamer: Latent / trajectory.
- I2A: Latent / e2e.
- MuZero: Latent / e2e.

## Two-agent Self-play
**Examples of self-play**
- Most two-agent board game-playing programs choose (versions of) themselves as opponent for simulation or learning.
- Minimax.
- Samuel’s checkers players.
- TD-Gammon: Tabula rasa self-play, shallow network, small alpha-beta search.

However, self-play is potentially unstable due to feedback and deadly triad. It is overcome in AlphaGo in dfferent ways.

### AlphaZero: Three Levels of Self-play
1. AlphaGo: The Champion.
2. AlphaGo Zero: Tabula rasa. The self-learner.
3. AlphaZero: Three games: Chess, Shogi, Go. The generalist.

#### Move-level Self-play
- Minimax: Assume you play best move, and opponent has your knowledge. Best of all actions.
- MCTS: Average of random playouts
    - Four operations: Selection, expansion, simulation, back-propagation.
    - Upper confidence bounds applied to trees (UCT) formula: UCT = winrate (exploitation) + $$C_p$$ * newness (exploration).

    $$
    \begin{align*}
    UCT(j) = \frac{w_i}{n_j} + C_p\sqrt{\frac{\ln n}{n_j}}
    \end{align*}
    $$

#### Example-level Self-play
- Learning: 
    - AlphaGo structure.
        - Four nets: Fast rollout policy, slow SL policy, slow RL policy, value net.
        - Three learning methods: 
            - Supervised small patterns fast rollout policy.
            - Supervised database grandmaster games.
            - Reinforcement from database de-correlated self-play games.
        - Policy $$\rightarrow$$ selection. Value $$\rightarrow$$ back-propagation. Playout $$\rightarrow$$ simulation.
   - AlphaGo Zero structure.
        - Zero-knowledge.
        - One net: ResNet with policy head and value head combined loss-function.
        - No random playout / game database.
        - One learning method: Self-play.
        - Tabula Rasa: Only the rules & input / output layers, zero heuristics, zero grandmaster games.
        - Faster: Curriculum learning.
        - Stable: Extra exploration / De-correlation.
            - How? MCTS + Noise + Exploration + Replay Buffer + Many games. AlphaGo Zero’s nets are not optimized against themselves, but against MCTS-improved versions of themselves.
   - AlphaZero structure.
       - Same net, same search, same tabula rasa self-play as AlphaGo Zero.
       -  Different input / output layers.
- Actor critic

#### Game-level Self-play
- Curriculum learning
    - Start with easy examples.
    - Many small steps are faster than one large step.
- Self transcending player

Curriculum learning & friends:
- Learning is generalization from example to example.
- Curriculum learning easy to hard concepts.
- Multi-task learning two tasks at the same time.
- Transfer learning from problem to problem.

## Multi-agent Reinforcement Learning
### Competition
- Zero sum: win / loss.
- Nash equilibrium.
    - The Nash equilibrium is point $$\pi^*$$ from which in a non-collaborative setting none of the agents has any incentive to deviate.
    - It is the optimal competitive strategy. Each agent chooses best actions for themselves assuming others do the same.
    - It is guaranteed to do no worse than tie against any opponent strategy.
    - For games of imperfect information the Nash equilibrium is an expected outcome.
- Counterfactual regret minimization (CFR).
    - Multi-agent, partial information, competition.
    - Iteratively minimize the regret of not having taken the right action, playing many "what-ifs" (counterfactuals).
    - CFR is probabilistic multi-agent version of competitive minimax.
    - Works quite well in Poker.

### Cooperation
- Non zero sum: win / win.
- Pareto front.
    - Pareto front is, in a cooperative setting, the combination of choices where no agent can be better off without at least making one other agent worse off.
    - It is the optimal cooperative strategy, the best outcome without hurting others.

**Cooperative behavior**
- Dealing with nonstationarity and partial observability can be done (ignored) by separate training, no communication.
- Realism can be improved with Centralized Training / Decentralized Execution. $$\Rightarrow$$ Centralized controller or interaction graphs.
- Overview of methods.
    - Value based: VDN, QMIX.
    - Policy based: COMA, MADDPG.
    - Opponent modeling: DRON, LOLA.
    - Communication: Diplomacy game.
    - Psychology: Heuristics.

### Mixed
- Prisoner’s dilemma.
- Iterated prisoner’s dilemma: Start being nice. Then, tit for tat.
- Emerging social norms.

### Algorithms
**Challenges**
- Partial observability: Large state space. Information sets.
- Nonstationary environments: Large state space. Calculate all configurations.
- Multiple agents: Large state space. Especially with simultaneous actions.

**Evolutionary approaches**
- Evolutionary algorithms.
- Swarm computing. e.g. ant colony optimization.
- Population based training.
    - Teams.
    - Hierarchical.
    - Cooperation / competition.
    - Within / between teams.
    - Blends RL and Evo.

## Hierarchical Reinforcement Learning
**Two types of abstraction**
- States: Representation learning is abstraction over states.
- Actions: Hierarchical RL is abstraction over actions.

**Macros**
- A primitive action is a regular, single-step, action.
- A macro-action is any multi-step action (sub-policy), such as: go from door A to door B.
- May be open-ended.

**Optimality**
- Note that the model-free small-step policy is likely better (more precise) than a policy incorporating a few large steps.
- HRL may be faster, but also coarser.

**Options**: Subpolicies of primitive actions.

### Algorithms
#### Tabular Algorithms
- STRIPS.
    - Planning system that controlled SHAKEY, the robot.
    - Macros were used to create higher level subroutines.
    - User defines subgoals and subroutines.
    - States are specified as conjunctions of predicates.
    - Actions are described as preconditions and effects.
    - Planning as search.
- HAM.
- MAXQ.
    - Hierarchical decomposition of MDP and value function.
    - Programmer defines subgoals and subpolicies.
    - MAXQ-Q-learning.
    - Introduce the Taxi domain.
- Abstraction hierarchies.
- Relation with planning, and thus with model-based.

Classical tabular methods suffer from combinatorial explosion of states/subgoals and actions/subpolicies for general methods.

#### Deep Learning Algorithms
- Feudal.
    - Hierarchical Q-learning of sub-managers learning to satisfy demands by managers.
    - Feudal Networks, using decoupled manager and worker modules, working at different time scales.
    - Manager computes latent state representation and goal vector.
    - LSTM.
    - Learning within the modules, to preserve local meaning.
    - Results show improvements over Flat RL (A3C).
- Option Critic.
    - Policy-gradient theorem for options, learn subpolicies and subgoals automatically.
    - Number of options is hyperparameter.
    - Good results on some ALE
- STRAW (Strategic Attentive Writer for Learning Macro-Actions).
    - Learn implicit plans from environment.
    - PacMan, Frostbite.
    - Text problems.
- HIRO.
    - Data efficient hierarchical reinforcement learning.
    - Sample efficient. Find subgoals and subpolicies.
    - Compute upper and goal-conditioned lower levels in parallel.
    - Off-policy (unstable lower levels).
    - MuJoCo
- HAC.
    - Learning multi-level hierarchies with hindsight.
    - Overcomes instability of joint learning of upper and lower levels.
- AMIGo.
    - Adversarially motivated intrinsic goals.
    - Teacher should learn appropriate tasks for student, not too hard, not too easy.
- Intrinsic IMGEP.

## References
1. Slides of Reinforcement Learning course, 2023 Spring, Leiden University.
2. Plaat, Aske. *Deep Reinforcement Learning*. 2022.
