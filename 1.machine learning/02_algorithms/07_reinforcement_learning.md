# Reinforcement Learning

## Quick Reference

| Concept | One-line definition |
|---------|-------------------|
| Agent | Entity that takes actions |
| Environment | World the agent interacts with |
| State (s) | Current situation |
| Action (a) | What the agent does |
| Reward (r) | Scalar feedback signal |
| Policy (π) | Mapping from state → action |
| Value function (V) | Expected cumulative reward from state s |
| Q-function (Q) | Expected cumulative reward from (s, a) pair |
| Episode | One full run from start to terminal state |
| Discount factor (γ) | How much to value future rewards (0-1) |

---

## Core Concepts

### The RL Loop

```
         ┌─────────────────────┐
         │     Environment     │
         │  state s_t ─────────┼──────────────┐
         │  reward r_t ────────┼──────────┐   │
         └─────────────────────┘          │   │
                                          ▼   ▼
                               ┌──────────────────────┐
                               │         Agent        │
                               │  policy π: s → a     │
                               │  update π from exp.  │
                               └──────────┬───────────┘
                                          │ action a_t
                               ┌──────────▼───────────┐
                               │     Environment      │
                               │  transition: s_{t+1} │
                               │    = T(s_t, a_t)     │
                               │  reward: r_{t+1}     │
                               │    = R(s_t, a_t, s') │
                               └──────────────────────┘
```

**The goal:** find policy π* that maximizes expected cumulative discounted reward:

```
G_t = r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + ... = Σ_{k=0}^∞ γ^k · r_{t+k+1}
```

### Markov Decision Process (MDP)

RL problems are formalized as MDPs. An MDP is a 5-tuple:

```
MDP = (S, A, T, R, γ)

S = state space (all possible states)
A = action space (all possible actions)
T = transition function: P(s' | s, a) = probability of reaching s' from s via a
R = reward function: R(s, a, s') = scalar reward for transition
γ = discount factor ∈ [0, 1)
```

**Markov property:** The future depends only on the current state, not history.

```
P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)
```

**Example MDP — Inventory Management:**

```
States:   s ∈ {0, 1, 2, ..., 10}  (units in stock)
Actions:  a ∈ {0, 1, 2, ..., 5}   (units to order)
Reward:   r = revenue - holding_cost - ordering_cost
Discount: γ = 0.95 (value future profits slightly less)

Transition: s_{t+1} = max(0, s_t + a_t - demand_t)
            demand_t = Poisson(λ=3)
```

---

## Value Functions

**State value function V^π(s):** Expected cumulative reward starting from state s, following policy π:

```
V^π(s) = E_π[G_t | s_t = s]
        = E_π[r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + ... | s_t = s]
```

**Action-value function (Q-function) Q^π(s, a):** Expected cumulative reward from state s, taking action a, then following π:

```
Q^π(s, a) = E_π[G_t | s_t = s, a_t = a]
           = E_π[r_{t+1} + γ·r_{t+2} | s_t = s, a_t = a]

Relationship:
  V^π(s) = Σ_a π(a|s) · Q^π(s, a)
  Q^π(s, a) = R(s,a) + γ · Σ_{s'} P(s'|s,a) · V^π(s')

Optimal policy: π(s) = argmax_a Q(s, a)
```

### Bellman Equations

The value functions satisfy recursive equations (Bellman equations):

**Bellman expectation equation:**
```
V^π(s) = Σ_a π(a|s) · [R(s,a) + γ · Σ_{s'} P(s'|s,a) · V^π(s')]
```

**Bellman optimality equation:**
```
V*(s) = max_a [R(s,a) + γ · Σ_{s'} P(s'|s,a) · V*(s')]
Q*(s,a) = R(s,a) + γ · Σ_{s'} P(s'|s,a) · max_{a'} Q*(s', a')
```

**Intuition:** The value of a state = immediate reward + discounted value of the next state.

---

## Q-Learning (Model-Free, Off-Policy)

**When to use:** You don't have a model of the environment (can't compute T or R explicitly). Must learn from trial and error.

**Key idea:** Maintain a table Q(s, a) for all state-action pairs. Update it using sampled experience via the Bellman equation.

### Q-Learning Update Rule

```
Q(s, a) ← Q(s, a) + α · [target - Q(s, a)]

target = r + γ · max_{a'} Q(s', a')

Full update:
Q(s, a) ← Q(s, a) + α · [r + γ · max_{a'} Q(s', a') - Q(s, a)]
                          └──────────────────────────────────────┘
                                       TD error (δ)
```

**Parameters:** α = learning rate (0.1 typical) · γ = discount factor (0.9-0.99) · ε = exploration rate for ε-greedy policy

### Dry Run — GridWorld Example

```
4×4 Grid:
  S = start (0,0)
  G = goal  (3,3) → reward +10
  H = hole  (1,1) → reward -10 (terminal)
  All other moves = reward -1 (step cost)

Actions: up, down, left, right
γ = 0.9,  α = 0.1,  ε = 0.1

Initial Q-table: all zeros (4×4 grid = 16 states, 4 actions = 64 entries)

Episode 1, Step 1:
  s = (0,0)
  ε-greedy: with prob 0.1 → random action; with prob 0.9 → argmax Q(s,a) = 0 (all tied) → random
  Choose: action = right → s+ = (0,1), r = -1

  Update Q((0,0), right):
    target = r + γ · max_{a'} Q((0,1), a')
           = -1 + 0 · max(0, 0, 0, 0)
           = -1 + 0 = -1
    Q((0,0), right) = 0 + 0.1 · (-1 - 0)
                    = -0.1

Episode 1, Step 2:
  s = (0,1)
  Choose: action = right → s+ = (0,2), r = -1

  Update Q((0,1), right):
    target = -1 + 0.9 · max Q((0,2), a') = -1 + 0 = -1
    Q((0,1), right) = 0 + 0.1 · (-1 - 0) = -0.1

Episode 1, reaching goal (after several steps):
  s = (3,2), action = right → s+ = (3,3) = GOAL, r = +10

  Update Q((3,2), right):
    target = 10 + 0.9 · 0   ← (3,3) is terminal, Q=0
           = 10
    Q((3,2), right) = 0 + 0.1 · (10 - 0) = 1.0

After many episodes — Q-values near goal (converged):
  Q((3,2), right) = 9.0      ← one step from goal
  Q((3,1), right) = 8.1      ← + γ · 9.0
  Q((3,0), right) = 7.29     ← + γ · 8.1
  Q((2,2), right) = 8.1      ← + γ · 9.0 via (3,2)
  Q((2,2), down)  = 7.29     ← + γ · 8.1 via (3,2) one step later

Optimal policy: each state points toward (3,3), avoiding hole at (1,1)
```

### Q-Learning Code

```python
import numpy as np

# Environment: 4x4 GridWorld
GRID_SIZE = 4
n_states  = GRID_SIZE * GRID_SIZE
n_actions = 4  # 0=up, 1=down, 2=left, 3=right

# Rewards
rewards = np.full((GRID_SIZE, GRID_SIZE), -1.0)
rewards[3][3] =  10.0  # goal
rewards[1][1] = -10.0  # hole

def step(state, action, grid_size=4):
    r, c = state // grid_size, state % grid_size
    if action == 0:   r = max(0, r-1)          # up
    elif action == 1: r = min(grid_size-1, r+1) # down
    elif action == 2: c = max(0, c-1)           # left
    elif action == 3: c = min(grid_size-1, c+1) # right
    next_state = r * grid_size + c
    reward = rewards[r][c]
    done = (r == 3 and c == 3) or (r == 1 and c == 1)
    return next_state, reward, done

# Q-Learning
Q = np.zeros((n_states, n_actions))

alpha     = 0.1   # learning rate
gamma     = 0.9   # discount factor
epsilon   = 0.1   # exploration rate
n_episodes = 5000

episode_rewards = []

for episode in range(n_episodes):
    state = 0  # start at (0,0)
    total_reward = 0

    for step_num in range(100):   # max steps per episode
        # ε-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(n_actions)   # explore
        else:
            action = np.argmax(Q[state])             # exploit

        next_state, reward, done = step(state, action)

        # Q-learning update
        td_target = reward + gamma * np.max(Q[next_state]) * (1 - done)
        td_error  = td_target - Q[state, action]
        Q[state, action] += alpha * td_error

        state = next_state
        total_reward += reward

        if done:
            break

    episode_rewards.append(total_reward)

# Evaluate greedy policy
def extract_policy(Q):
    action_names = ['↑', '↓', '←', '→']
    policy = [[action_names[np.argmax(Q[r*4+c])] for c in range(4)] for r in range(4)]
    return policy

print("Learned policy:")
for row in extract_policy(Q):
    print(' '.join(row))
# Expected (approximately):
# → → → ↓
# → H ↓ ↓
# → → ↓ ↓
# → → → G
```

---

## SARSA (On-Policy TD Control)

Q-learning is **off-policy**: updates use max_{a'} Q(s', a') — the best possible action, regardless of what you'd actually do.

SARSA is **on-policy**: updates use Q(s', a') where a' is the action you actually take next.

```
SARSA update:
Q(s, a) ← Q(s, a) + α · [r + γ · Q(s', a') - Q(s, a)]
                                    ↑
                           actual next action (not max)
```

**Key difference:**
```
Q-Learning: safer "theoretical" policy — learns optimal Q regardless of exploration
SARSA:      learns Q for the policy it's actually following (including ε-greedy noise)

Example: cliff-walking
  Q-Learning: finds shortest path right along cliff edge (optimal but risky)
  SARSA:      finds safer path away from cliff (suboptimal distance, lower fall risk)

Rule of thumb:
  - Use Q-learning when you want the optimal policy at convergence
  - Use SARSA when exploration itself is costly/dangerous
```

---

## Policy Gradient (Model-Free, On-Policy)

Q-learning learns a value function and derives policy from it (indirect). Policy gradient methods learn the policy **directly**.

**Parameterize policy:** π_θ(a|s) = probability of taking action a in state s, parameterized by θ.

**Objective:** maximize expected return:

```
J(θ) = E_π[G_{t=0}] = E_π[Σ_{t=0}^∞ γ^t · r_{t+1}]
```

**Policy gradient theorem:**

```
∇_θ J(θ) = E_π[G_t · log π_θ(a_t|s_t) · G_t]
```

**Intuition:** If an action led to high return G_t, increase its probability. If low return, decrease it. The log π gradient tells us the direction to push θ.

### REINFORCE Algorithm (Monte Carlo Policy Gradient)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, n_states, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

policy    = PolicyNetwork(n_states=4, n_actions=2)  # e.g., CartPole
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
gamma     = 0.99

def compute_returns(rewards, gamma):
    """Compute discounted returns from a list of rewards."""
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    # Normalize returns (reduces variance)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

def train_episode(env):
    states, actions, rewards, log_probs = [], [], [], []

    state, _ = env.reset()
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state)
        probs  = policy(state_tensor)
        dist   = torch.distributions.Categorical(probs)
        action = dist.sample()

        next_state, reward, done, truncated, _ = env.step(action.item())
        done = done or truncated

        log_probs.append(dist.log_prob(action))
        rewards.append(reward)
        state = next_state

    # Compute returns
    returns = compute_returns(rewards, gamma)

    # Policy gradient loss
    # loss = -E[log π(a|s) · G_t]
    loss = -torch.stack([lp * R for lp, R in zip(log_probs, returns)]).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return sum(rewards)

# Training loop
import gymnasium as gym
env = gym.make("CartPole-v1")
for episode in range(1000):
    total_reward = train_episode(env)
    if episode % 100 == 0:
        print(f"Episode {episode}: reward = {total_reward:.1f}")
```

### REINFORCE Dry Run

```
CartPole episode:
  s0: a=right → r=1.0
  s1: a=left  → r=1.0
  s2: a=right → r=1.0
  s3: a=fell  → r=0.0  (terminal)

γ = 0.99

Discounted returns (from end):
  G3 = 0.0                          (terminal)
  G2 = 1.0 + 0.99 · 0.0 = 1.0
  G1 = 1.0 + 0.99 · 1.0 = 1.99
  G0 = 1.0 + 0.99 · 1.99 = 2.97

After normalization (mean=1.74, std=1.23):
  G0_norm = (2.97 - 1.74) / 1.23 = +1.00
  G1_norm = (1.99 - 1.74) / 1.23 = +0.20
  G2_norm = (1.00 - 1.74) / 1.23 = -0.60

Policy at s0: π(right|s0) = 0.55 → log π = log(0.55) = -0.598

Loss contribution at t=0:
  -log π(a|s0) · G0_norm = -(-0.598) · 1.00 = +0.598

Gradient update:
  T=0 loss = +0.598 → θ updates to slightly increase π(right|s0)
  (positive G_norm means "right" was a good choice here → increase its probability)

At t=2: G2_norm = -0.60 → loss = -log π(right|s2) · (-0.60) = negative
Gradient update: slightly decrease π(right|s2)
(negative return means "right" at s2 contributed to falling)
```

---

## Key Algorithms Comparison

| Algorithm | Type | Policy | Needs model? | Strengths | Weaknesses |
|-----------|------|--------|-------------|-----------|------------|
| Q-Learning | Value-based | Implicit (greedy over Q) | No | Simple, tabular, guaranteed convergence | Discrete actions only |
| SARSA | Value-based | On-policy | No | Safer exploration | Suboptimal w.r.t. optimal policy |
| REINFORCE | Policy gradient | Explicit π_θ | No | Continuous actions, stochastic policy | High variance, slow |
| Actor-Critic | Hybrid | π_θ + V_φ | No | Lower variance than REINFORCE | More complex |
| DQN | Deep value | Implicit (neural Q) | No | Scalable to large state spaces | Discrete only |
| PPO | Policy gradient | π_θ clipped | No | Stable, widely used in LLMs | Computationally heavy |

DQN, PPO, A3C covered in `../../2.deep learning/02_architectures/07_reinforcement_learning.md`.

For modern alignment algorithms (PPO for RLHF, DPO, GRPO, KTO, ORPO) see the deep treatment in `10_reinforcement_learning_deep.md` — this is the 2024-25 interview-relevant material.

---

## Exploration vs Exploitation

**The dilemma:** Should the agent use what it knows (exploit) or try new actions (explore)?

### ε-Greedy

```python
if random() < epsilon:
    action = random_action()       # explore
else:
    action = argmax(Q[state])      # exploit

# Decay epsilon over time:
epsilon = max(0.01, epsilon * 0.995)  # starts at 1.0, decays to 0.01
```

### Upper Confidence Bound (UCB)

```
action = argmax[ Q(s,a) + c · √(ln(t) / N(s,a)) ]

N(s,a) = number of times action a was taken in state s
t      = total number of steps
c      = exploration constant

Intuition: try actions that are either high-value OR rarely tried (high uncertainty)
```

### Thompson Sampling

Maintain a distribution over Q values. Sample from the distribution, pick best action under sample. Bayesian approach — naturally balances explore/exploit.

---

## When to Use RL (vs Supervised Learning)

| Use RL when... | Use Supervised Learning when... |
|---------------|--------------------------------|
| No labeled data; only reward signal | Labeled (state → action) pairs available |
| Agent must learn from interaction | Data is static, collected offline |
| Environment has delayed rewards | Each input has an immediate correct output |
| Sequential decision-making matters | i.i.d. predictions |
| Examples: games, robotics, trading | Examples: classification, regression |

**RL applied to LLMs:** RLHF (Reinforcement Learning from Human Feedback): policy = LLM, reward = human preference score — PPO updates LLM weights to maximize reward while staying close to original policy (KL penalty). See `../../6.llms/03b_alignment_end_to_end.md` for full trace. For DPO, GRPO, KTO, ORPO — the **reward-model-free** alternatives that have largely replaced PPO in 2024-25 production RLHF — see `10_reinforcement_learning_deep.md` and `../../6.llms/06_alignment_follow_ups.md`.

---

## Gotchas

**Reward shaping.** Adding intermediate rewards to speed up learning is powerful but dangerous. Poorly shaped rewards lead to unexpected behavior ("reward hacking"). Always check if your agent optimizes the true objective, not a proxy.

**Discount factor γ.** γ=1.0 gives undiscounted cumulative reward (good for episodic tasks). γ<1 required for infinite-horizon tasks to keep G_t finite. γ=0.99 is a common default.

**State representation matters.** Raw pixels → need CNNs (DQN). Tabular Q-learning only works for small, discrete state spaces. Most real problems need function approximation (neural networks).

**Sample efficiency.** RL is notoriously sample-inefficient. Q-learning on Atari requires ~50M environment steps. Compare: a child learns Pong in minutes. Off-policy methods (DQN, Q-learning) are more sample-efficient than on-policy (REINFORCE, PPO) because they can reuse past experience (replay buffer).

**Convergence guarantees.** Tabular Q-learning converges to Q* under: every (s,a) visited infinitely often, learning rate α decays properly. Neural network Q-learning (DQN) — no convergence guarantee, but works in practice.

---

## Interview Q&A

**Q: What is the difference between model-based and model-free RL?**
A: Model-based RL has or learns a model of the environment (transition function T and reward function R), and uses it for planning (e.g., value iteration, AlphaGo Monte Carlo Tree Search). Model-free RL (Q-learning, PPO) learns directly from interaction without modeling the environment. Model-based is more sample-efficient but requires an accurate model; model-free is more general and doesn't suffer from model errors.

**Q: Why does Q-learning use max_{a'} Q(s', a') but SARSA uses Q(s', a')?**
A: Q-learning is off-policy — it updates toward the best possible next action, giving the Bellman optimality target. SARSA is on-policy — it updates toward the action the current policy would actually take (including exploration). Q-learning learns Q* (optimal); SARSA learns Q^π (value under current policy). In cliff-walking: Q-learning finds the shortest path (optimal), SARSA avoids the cliff (safer given exploration).

**Q: What is the policy gradient theorem?**
A: ∇_θ J(θ) = E_π[V_θ·∇_θ log π_θ(a_t|s_t) · G_t]. It says: to increase expected return, nudge policy parameters in the direction that increases the log probability of actions that led to high returns, and decreases it for low-return actions. The log trick avoids needing to differentiate through the environment.

**Q: What is the credit assignment problem?**
A: When a reward is received at time t, it's unclear which past actions caused it. In a game of chess, the winning move might be 50 steps before the reward. Solutions: discounting (recent rewards weighted more), eligibility traces (track contribution of each (s,a) over time), attention/transformer architectures for long sequences.

**Q: How is RL used in training LLMs?**
A: RLHF (Reinforcement Learning from Human Feedback). The LLM is the policy π_θ. A reward model R_φ is trained on human preference comparisons. PPO updates the LLM to maximize R_φ(s,a) while adding a KL penalty: R_total = R_φ - β·KL(π_θ || π_ref) to prevent the model from deviating too far from the original pre-trained model.

---

## Connections

- Deep RL (DQN, PPO, A3C): `../../2.deep learning/02_architectures/07_reinforcement_learning.md`
- Modern alignment (PPO for RLHF, DPO, GRPO, KTO, ORPO): `10_reinforcement_learning_deep.md` — 2024-25 interview material
- RLHF in LLMs: `../../6.llms/03b_alignment_end_to_end.md`
- Alignment follow-ups: `../../6.llms/06_alignment_follow_ups.md` — DPO, KTO, ORPO, IPO comparison
- Policy Gradient math: relies on expectation/log-derivative trick from probability theory
- Markov chains: MDP is a controlled Markov chain; stationary distributions connect to value convergence
- Exploration as Bayesian optimization: Thompson sampling and UCB are the same ideas seen in `04_probabilistic.md` Bayesian Optimization section

---

## Key Takeaway

RL = agent learns by interacting with environment to maximize cumulative reward. Core framework: MDP (S, A, T, R, γ). Two main families: **value-based** (Q-learning — learn Q(s,a), pick greedy action) and **policy gradient** (REINFORCE, PPO — directly optimize π_θ). Q-learning update: Q(s,a) ← Q(s,a) + α·[r + γ·max Q(s',a') - Q(s,a)]. Policy gradient update: ∇_θ J = E[∇θ log π · G_t]. Deep RL (DQN, PPO) scales these to large state spaces using neural networks — covered in the deep learning folder.
