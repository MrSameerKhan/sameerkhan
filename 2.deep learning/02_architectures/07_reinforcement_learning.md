# Deep Reinforcement Learning

> **Prerequisites:** RL fundamentals (MDP, Q-learning, Policy Gradient) → `1.machine learning/02_algorithms/07_reinforcement_learning.md`  
> **This file:** scaling RL with neural networks — DQN, A3C, PPO, and RLHF.

---

## Why Deep RL?

Tabular Q-learning stores Q(s, a) for every (state, action) pair.  
**Problem:** Atari has ~10^33,000 possible screen states. A table is impossible.

**Solution:** Replace the Q-table with a neural network:
```
Q(s, a; θ) ≈ Q*(s, a)

Input:  state s (e.g., 84×84×4 grayscale frames)
Output: Q-values for all actions simultaneously
         [Q(s, left), Q(s, right), Q(s, fire)] → shape (n_actions,)
```

---

## DQN — Deep Q-Network

**Paper:** "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)  
**Key insight:** Two tricks make neural Q-learning stable: **experience replay** + **target network**.

### Architecture

```
Input: 4 stacked 84×84 grayscale frames → (4, 84, 84)
       ↓
Conv2D(32, 8×8, stride=4) → (32, 20, 20)   ReLU
       ↓
Conv2D(64, 4×4, stride=2) → (64, 9, 9)     ReLU
       ↓
Conv2D(64, 3×3, stride=1) → (64, 7, 7)     ReLU
       ↓
Flatten → 3136
       ↓
Linear(3136, 512)                           ReLU
       ↓
Linear(512, n_actions)                      no activation
       ↓
Output: [Q(s, a₁), Q(s, a₂), ..., Q(s, aₙ)]  e.g., 18 actions for Atari
```

### Problem 1: Correlated Samples → Experience Replay

**Problem:** Sequential frames (s_t, s_{t+1}) are highly correlated. Training a neural net on correlated data → non-i.i.d. → unstable training, catastrophic forgetting.

**Solution:** Store transitions in a **replay buffer**. Sample random mini-batches.

```python
from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=32):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)
```

### Problem 2: Moving Targets → Target Network

**Problem:** DQN update:
```
Loss = [Q(s,a; θ) - (r + γ · max_{a'} Q(s', a'; θ))]²
```
Both the prediction Q(s,a; θ) and the target Q(s',a'; θ) use the same weights θ.  
Every gradient step changes the target → chasing a moving target → divergence.

**Solution:** Maintain a **target network** θ⁻ (frozen copy). Update only every C steps.

```
target = r + γ · max_{a'} Q(s', a'; θ⁻)   ← frozen weights
Loss   = [Q(s,a; θ) - target]²              ← update θ only

Every C=1000 steps: θ⁻ ← θ   (hard update)
OR every step:      θ⁻ ← τ·θ + (1-τ)·θ⁻   (soft update, τ=0.005)
```

### Full DQN Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 128), nn.ReLU(),
            nn.Linear(128, 128),  nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x):
        return self.net(x)

# Hyperparameters
BATCH_SIZE   = 64
GAMMA        = 0.99
LR           = 1e-4
EPSILON_START = 1.0
EPSILON_END   = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 100       # steps between target network updates
BUFFER_SIZE   = 50_000
MIN_REPLAY    = 1_000     # start training after this many transitions

env = gym.make("CartPole-v1")
n_obs     = env.observation_space.shape[0]  # 4
n_actions = env.action_space.n              # 2

q_net      = DQN(n_obs, n_actions)
target_net = DQN(n_obs, n_actions)
target_net.load_state_dict(q_net.state_dict())
target_net.eval()

optimizer = optim.Adam(q_net.parameters(), lr=LR)
buffer    = ReplayBuffer(BUFFER_SIZE)
epsilon   = EPSILON_START
step_count = 0

for episode in range(500):
    state, _ = env.reset()
    total_reward = 0

    for _ in range(500):
        # ε-greedy action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_net(torch.FloatTensor(state))
                action = q_values.argmax().item()

        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        buffer.push(state, action, reward, next_state, float(done))
        state = next_state
        total_reward += reward
        step_count += 1

        # Train
        if len(buffer) >= MIN_REPLAY:
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

            states_t      = torch.FloatTensor(states)
            next_states_t = torch.FloatTensor(next_states)
            actions_t     = torch.LongTensor(actions)
            rewards_t     = torch.FloatTensor(rewards)
            dones_t       = torch.FloatTensor(dones)

            # Current Q values
            q_current = q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

            # Target Q values (use target network, no gradient)
            with torch.no_grad():
                q_next   = target_net(next_states_t).max(1)[0]
                q_target = rewards_t + GAMMA * q_next * (1 - dones_t)

            loss = nn.MSELoss()(q_current, q_target)
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping (important for stability)
            nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10)
            optimizer.step()

            # Update target network
            if step_count % TARGET_UPDATE == 0:
                target_net.load_state_dict(q_net.state_dict())

        if done:
            break

    # Decay epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    if episode % 50 == 0:
        print(f"Episode {episode:4d} | Reward: {total_reward:6.1f} | ε: {epsilon:.3f}")
```

### DQN Dry Run (Loss Computation)

```
Mini-batch of 3 transitions:
  (s₀, right, r=1.0, s₁, done=False)
  (s₁, left,  r=1.0, s₂, done=False)
  (s₂, right, r=0.0, s₃, done=True)

Current Q-network output:
  Q(s₀) = [2.1, 1.8]     → Q(s₀, right) = 1.8  (action=1)
  Q(s₁) = [1.5, 2.2]     → Q(s₁, left)  = 1.5  (action=0)
  Q(s₂) = [0.9, 1.1]     → Q(s₂, right) = 1.1  (action=1)

Target network output:
  Q⁻(s₁) = [1.4, 2.0]   → max = 2.0
  Q⁻(s₂) = [0.8, 1.0]   → max = 1.0
  Q⁻(s₃) = [0.1, 0.2]   → max = 0.2  (but done=True → ignored)

Targets (γ=0.99):
  y₀ = 1.0 + 0.99 · 2.0 · (1-0) = 1.0 + 1.98 = 2.98
  y₁ = 1.0 + 0.99 · 1.0 · (1-0) = 1.0 + 0.99 = 1.99
  y₂ = 0.0 + 0.99 · 0.2 · (1-1) = 0.0          ← done=True, future term = 0

TD errors:
  δ₀ = y₀ - Q(s₀, right) = 2.98 - 1.8 = 1.18
  δ₁ = y₁ - Q(s₁, left)  = 1.99 - 1.5 = 0.49
  δ₂ = y₂ - Q(s₂, right) = 0.00 - 1.1 = -1.10

MSE Loss = (1.18² + 0.49² + 1.10²) / 3
         = (1.392 + 0.240 + 1.210) / 3
         = 2.842 / 3
         = 0.947
```

### DQN Variants

| Variant | Key improvement |
|---------|----------------|
| Double DQN | Decouple action selection (θ) from evaluation (θ⁻) → reduces overestimation |
| Dueling DQN | Separate value V(s) and advantage A(s,a) streams → Q(s,a) = V(s) + A(s,a) |
| Prioritized Replay | Sample transitions by TD error magnitude → focus on surprising transitions |
| Rainbow DQN | All of the above + multi-step returns + distributional RL → SOTA on Atari |

**Double DQN fix:**
```
Standard DQN:  target = r + γ · max_{a'} Q(s', a'; θ⁻)
Double DQN:    target = r + γ · Q(s', argmax_{a'} Q(s', a'; θ); θ⁻)
                                         └─select with θ─┘  └─eval with θ⁻─┘
```

---

## A3C — Asynchronous Advantage Actor-Critic

**Paper:** "Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al., 2016)

**Key idea:** Run multiple agents in parallel with their own environment copies. Each worker computes gradients and asynchronously updates a shared global network.

### Architecture

```
Global network (shared weights θ_global):
  Actor:  π_θ(a|s)  → action probabilities
  Critic: V_θ(s)    → state value estimate

Worker threads (n=8 or 16):
  1. Copy global weights: θ_local ← θ_global
  2. Play for t_max=5 steps (or until episode end)
  3. Compute gradients on local data
  4. Apply gradients to θ_global (async, no lock)
```

### Actor-Critic Loss

```
Advantage:
  A(s_t, a_t) = R_t - V(s_t; θ)
  R_t = r_t + γ·r_{t+1} + ... + γ^{T-1}·r_{T-1} + γ^T·V(s_T; θ)
  (n-step return — better credit assignment than single-step)

Actor loss:  L_actor  = -log π(a_t|s_t) · A(s_t, a_t)
Critic loss: L_critic = (R_t - V(s_t))²
Entropy:     H = -Σ_a π(a|s)·log π(a|s)  ← encourages exploration

Total loss: L = L_actor + 0.5·L_critic - 0.01·H
```

**Why advantage A instead of G_t?**
```
REINFORCE:      ∇ log π · G_t          ← high variance (G_t varies a lot)
Actor-Critic:   ∇ log π · A(s,a)       ← lower variance (A is relative, not absolute)

A > 0: this action was better than expected → increase its probability
A < 0: this action was worse than expected  → decrease its probability
A ≈ 0: this action was as expected          → no update
```

### A3C Code (Simplified)

```python
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim
import gymnasium as gym
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_obs, 128), nn.ReLU()
        )
        self.actor  = nn.Linear(128, n_actions)   # policy head
        self.critic = nn.Linear(128, 1)           # value head

    def forward(self, x):
        features = self.shared(x)
        logits   = self.actor(features)           # raw scores
        value    = self.critic(features).squeeze(-1)
        return logits, value

    def get_action(self, state):
        logits, value = self.forward(torch.FloatTensor(state))
        dist   = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

def worker(global_model, optimizer, worker_id, n_steps=5, gamma=0.99):
    env = gym.make("CartPole-v1")
    local_model = ActorCritic(4, 2)

    for episode in range(500):
        local_model.load_state_dict(global_model.state_dict())

        state, _ = env.reset()
        log_probs, values, rewards, dones = [], [], [], []

        for _ in range(n_steps):
            action, log_prob, value = local_model.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            dones.append(done)

            state = next_state
            if done:
                state, _ = env.reset()
                break

        # Compute n-step returns
        R = 0.0 if dones[-1] else local_model.forward(torch.FloatTensor(state))[1].item()
        returns = []
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + gamma * R * (1 - done)
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        values  = torch.stack(values)

        # Losses
        advantages = returns - values.detach()
        actor_loss  = -(torch.stack(log_probs) * advantages).mean()
        critic_loss = nn.MSELoss()(values, returns)
        entropy     = -torch.stack([lp.exp() * lp for lp in log_probs]).mean()
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        # Async gradient update to global model
        optimizer.zero_grad()
        loss.backward()
        # Copy gradients to global model
        for local_p, global_p in zip(local_model.parameters(), global_model.parameters()):
            global_p._grad = local_p.grad
        optimizer.step()
```

---

## PPO — Proximal Policy Optimization

**Paper:** "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)  
**Why it dominates:** Simple to implement, stable, parallelizable. Default algorithm in OpenAI Five, ChatGPT RLHF, most robotics work.

### The Problem with Policy Gradient

Standard policy gradient (REINFORCE, A3C) can take huge steps that destroy the policy:
```
If a lucky trajectory gives G_t = 100 for some rare action,
gradient step strongly increases π(a|s) for all states where action a was taken.
Next rollout: policy is now weird, worse trajectories, gradient overcorrects back.
→ Training instability: policy oscillates, doesn't converge.
```

**Solution:** Limit how much the policy changes in each update.

### TRPO vs PPO

**TRPO** (Trust Region Policy Optimization): adds a KL constraint:
```
maximize E[r_t(θ) · Â_t]
subject to: E[KL(π_old || π_new)] ≤ δ

r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  ← probability ratio
```
Requires second-order optimization (conjugate gradient). Complex, slow.

**PPO** approximates TRPO with a **clipped objective** — simpler, first-order, similar performance.

### PPO Clipped Objective

```
r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)   ← probability ratio

L_CLIP(θ) = E[min(
    r_t(θ) · Â_t,                            ← standard policy gradient
    clip(r_t(θ), 1-ε, 1+ε) · Â_t            ← clipped version
)]

ε = 0.2 typically (ratio stays in [0.8, 1.2])
```

**Intuition:**
```
If Â_t > 0 (good action):
  We want to increase π(a|s) → r_t(θ) > 1
  But we cap the gain at r_t = 1+ε → prevents taking too large a step

If Â_t < 0 (bad action):
  We want to decrease π(a|s) → r_t(θ) < 1
  But we floor at r_t = 1-ε → prevents catastrophic decrease

The min() ensures we take the pessimistic (conservative) bound.
```

**Dry run:**
```
Old policy: π_old(left|s) = 0.3
New policy: π_new(left|s) = 0.5
r_t = 0.5 / 0.3 = 1.667   ← big increase

ε = 0.2 → clip range [0.8, 1.2]
clip(1.667, 0.8, 1.2) = 1.2

Assume Â_t = 2.0 (this was a good action):
  Unclipped: 1.667 × 2.0 = 3.333
  Clipped:   1.2   × 2.0 = 2.4

L_CLIP = min(3.333, 2.4) = 2.4    ← clipped value wins
Gradient from L_CLIP will increase π(left|s), but less aggressively than unclipped.
```

### PPO Full Loss

```
L_total = L_CLIP - c₁·L_VF + c₂·H

L_VF = (V_θ(s) - R_t)²              ← value function loss (critic)
H    = -Σ_a π_θ(a|s)·log π_θ(a|s)  ← entropy bonus (encourages exploration)
c₁ = 0.5, c₂ = 0.01 (typical)
```

### PPO Code

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_obs, 64), nn.Tanh(),
            nn.Linear(64, 64),   nn.Tanh()
        )
        self.actor  = nn.Linear(64, n_actions)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        f = self.shared(x)
        return torch.distributions.Categorical(logits=self.actor(f)), self.critic(f).squeeze(-1)

class PPO:
    def __init__(self, n_obs, n_actions):
        self.model     = ActorCritic(n_obs, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        self.clip_eps  = 0.2
        self.gamma     = 0.99
        self.lam       = 0.95  # GAE lambda
        self.n_epochs  = 4     # update epochs per rollout
        self.batch_size = 64

    def collect_rollout(self, env, n_steps=2048):
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        state, _ = env.reset()

        for _ in range(n_steps):
            state_t = torch.FloatTensor(state)
            with torch.no_grad():
                dist, value = self.model(state_t)
            action = dist.sample()

            next_state, reward, done, truncated, _ = env.step(action.item())
            done = done or truncated

            states.append(state_t)
            actions.append(action)
            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(reward)
            dones.append(done)

            state = next_state if not done else env.reset()[0]

        # Compute GAE advantages
        advantages = self._compute_gae(rewards, values, dones)
        returns    = advantages + torch.stack(values)

        return (torch.stack(states), torch.stack(actions),
                torch.stack(log_probs), advantages, returns)

    def _compute_gae(self, rewards, values, dones):
        """Generalized Advantage Estimation — lower variance than MC returns."""
        gae = 0
        advantages = []
        next_value = 0  # after last step
        for r, v, done in zip(reversed(rewards), reversed(values), reversed(dones)):
            delta = r + self.gamma * next_value * (1 - done) - v.item()
            gae   = delta + self.gamma * self.lam * (1 - done) * gae
            advantages.insert(0, gae)
            next_value = v.item()
        advantages = torch.FloatTensor(advantages)
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def update(self, states, actions, old_log_probs, advantages, returns):
        n = len(states)
        for _ in range(self.n_epochs):
            # Random mini-batches from rollout
            indices = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = indices[start:start + self.batch_size]
                batch_states    = states[idx]
                batch_actions   = actions[idx]
                batch_old_lp    = old_log_probs[idx]
                batch_adv       = advantages[idx]
                batch_returns   = returns[idx]

                dist, values = self.model(batch_states)
                new_log_probs = dist.log_prob(batch_actions)
                entropy       = dist.entropy().mean()

                # Clipped ratio
                ratio    = (new_log_probs - batch_old_lp).exp()
                surr1    = ratio * batch_adv
                surr2    = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values, batch_returns)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

# Training
env = gym.make("CartPole-v1")
agent = PPO(n_obs=4, n_actions=2)

for iteration in range(50):
    states, actions, log_probs, advantages, returns = agent.collect_rollout(env)
    agent.update(states, actions, log_probs, advantages, returns)
    print(f"Iteration {iteration}")
```

---

## RLHF — Reinforcement Learning from Human Feedback

**Where you'll see this in interviews:** "How does ChatGPT get trained?" or "How does RLHF work?"

### Three-Stage Pipeline

```
Stage 1: Supervised Fine-Tuning (SFT)
  - Start with pre-trained LLM
  - Fine-tune on (prompt, human-written response) pairs
  - Standard cross-entropy loss
  - Output: SFT model π_SFT

Stage 2: Reward Model Training
  - Collect comparison data: show human two responses A, B, ask "which is better?"
  - Train reward model: R_φ(prompt, response) → scalar score
  - Architecture: LLM + linear head on [EOS] token
  - Loss: -E[log σ(R_φ(y_w) - R_φ(y_l))]  (y_w = preferred, y_l = rejected)
  - Output: reward model R_φ

Stage 3: PPO Fine-Tuning
  - Policy: π_θ = SFT model, initialized from π_SFT
  - For each prompt x, generate response y ~ π_θ(·|x)
  - Reward: r = R_φ(x, y) - β · KL(π_θ(·|x) || π_SFT(·|x))
              └─── quality ───┘   └────── don't drift too far ──────┘
  - Use PPO to maximize r w.r.t. θ
  - β controls how much to penalize deviation from SFT model (β ≈ 0.1–0.2)
```

**KL penalty intuition:**
```
Without KL penalty:
  PPO maximizes reward at all costs → model outputs gibberish that fools reward model
  (reward hacking)

With KL penalty:
  r_total = R_φ(x,y) - β · KL(π_θ || π_SFT)
  Model must balance: make reward model happy AND stay close to original SFT model
  β=0 → full RL, can drift arbitrarily
  β→∞ → no update, stays at SFT
  β=0.1-0.2 → sweet spot for alignment
```

### RLHF Reward Model

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Reward model: take LLM, add value head
class RewardModel(nn.Module):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        from transformers import GPT2Model
        self.model = GPT2Model.from_pretrained(model_name)
        self.value_head = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Use last non-padding token (EOS position)
        last_token_hidden = outputs.last_hidden_state[:, -1, :]
        reward = self.value_head(last_token_hidden).squeeze(-1)
        return reward

def reward_model_loss(reward_chosen, reward_rejected):
    """Bradley-Terry preference model loss."""
    # P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
    # Maximize log P → minimize -log σ(r_chosen - r_rejected)
    return -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()
```

---

## Comparison: DQN vs A3C vs PPO

| | DQN | A3C | PPO |
|--|-----|-----|-----|
| Policy type | Off-policy | On-policy | On-policy |
| Action space | Discrete only | Both | Both |
| Parallelism | Replay buffer | Async workers | Vectorized envs |
| Stability | Target network + replay | Moderate | High (clipping) |
| Sample efficiency | High (replay) | Low | Medium |
| Implementation | Moderate | Complex (async) | Simple |
| Use case | Atari-style games | CPU-based parallel | General, RLHF |

---

## Key Numbers

| Algorithm | Atari training steps | Wall-clock (1 GPU) |
|-----------|---------------------|-------------------|
| DQN | 50M | ~3 days |
| A3C (16 workers) | 200M | ~4 hours (CPU parallel) |
| PPO (vectorized) | 10M | ~1 day |
| Rainbow DQN | 200M | ~3 days |

---

## Gotchas

**PPO clip ε too small.** ε=0.1 → too conservative, slow learning. ε=0.3 → too large, unstable. ε=0.2 is the empirically validated default.

**Advantage normalization.** Always normalize advantages within a mini-batch (subtract mean, divide by std). Un-normalized advantages have wildly different scales across environments → learning rate sensitivity.

**GAE lambda.** λ=1 → Monte Carlo returns (unbiased, high variance). λ=0 → TD(0) (biased, low variance). λ=0.95 is the standard sweet spot.

**Replay buffer size.** Too small → correlated samples. Too large → includes very old, off-policy transitions that hurt learning. 1M transitions is standard for Atari DQN.

**RLHF reward hacking.** The reward model is imperfect. PPO will find edge cases that the reward model scores highly but humans don't. Solution: KL penalty, early stopping, iterative reward model updates (RLHF loop runs 2-3 times in practice).

---

## Interview Q&A

**Q: Why does DQN use a target network?**
A: Without it, both the prediction Q(s,a;θ) and the target r+γ·max Q(s',a';θ) use the same weights. Every gradient step moves the target, creating a moving target problem — the network never settles. The target network θ⁻ is a frozen copy of θ, updated every C steps. This makes the target stationary within a window, stabilizing training significantly.

**Q: What problem does PPO's clipping solve?**
A: Standard policy gradient can take large steps that collapse the policy. If a high-advantage trajectory causes a big parameter update, the resulting policy may be completely different and start generating bad rollouts. PPO clips the probability ratio r_t(θ) to [1-ε, 1+ε], preventing any single update from changing the policy too dramatically. This makes training stable without requiring expensive second-order methods like TRPO.

**Q: What is GAE and why use it instead of Monte Carlo returns?**
A: GAE (Generalized Advantage Estimation) computes advantages as an exponentially-weighted average of n-step TD errors: A_t = Σ_{l=0}^∞ (γλ)^l · δ_{t+l}. λ=1 → same as Monte Carlo (unbiased but high variance from noisy long-horizon rewards). λ=0 → TD(0) (low variance but biased). λ=0.95 interpolates — lower variance than MC, lower bias than TD(0).

**Q: Explain RLHF in 3 steps.**
A: (1) SFT: fine-tune base LLM on human-written (prompt, response) pairs. (2) Reward model: train a model to predict human preference scores from comparison data. (3) PPO: treat the LLM as a policy, use PPO to maximize the reward model's score with a KL penalty to prevent reward hacking (the policy must stay close to the SFT model).

**Q: When would you use DQN vs PPO?**
A: DQN: discrete action spaces (Atari, board games), high replay buffer efficiency needed. PPO: continuous action spaces (robotics), stable training needed (default choice), RLHF fine-tuning of LLMs. PPO is now the default RL algorithm in most practical applications — simpler than A3C, more stable than vanilla policy gradient, works for both discrete and continuous.

---

## Connections

- **RL fundamentals (MDP, Q-learning, REINFORCE):** `1.machine learning/02_algorithms/07_reinforcement_learning.md`
- **RLHF alignment details:** `6.llms/10_alignment_end_to_end.md`
- **CNNs used in DQN for Atari:** `2.deep learning/02_architectures/02_cnn.md`
- **Transformers in RL (Decision Transformer):** `5.transformers/02_models/`

## Key Takeaway

Deep RL = RL + neural networks to handle large state spaces. Three core algorithms: **DQN** (Q-table → neural net, stabilized via replay buffer + target network), **A3C** (actor-critic, async parallel workers, advantage reduces variance), **PPO** (on-policy actor-critic with clipped objective — prevents catastrophic policy updates, default algorithm for RLHF). The progression: Q-learning (tabular) → DQN (neural Q, discrete) → A3C (neural actor-critic, parallel) → PPO (clipped, stable, universal). PPO is the go-to: it powers ChatGPT's RLHF, OpenAI Five, and most modern robotics work.
