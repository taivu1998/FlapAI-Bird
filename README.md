# FlapAI Bird

**Training AI Agents to Play Flappy Bird with Reinforcement Learning**

[![Paper](https://img.shields.io/badge/arXiv-2003.09579-b31b1b.svg)](https://arxiv.org/abs/2003.09579)
[![Poster](https://img.shields.io/badge/Stanford-CS221_Poster-8C1515.svg)](https://stanford-cs221.github.io/autumn2019-extra/posters/18.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive reinforcement learning framework that implements and benchmarks six progressively sophisticated agents for the Flappy Bird environment — from a stochastic baseline to a deep Q-network with convolutional feature extraction and experience replay. After training, our best agents achieve **superhuman performance with scores exceeding 2,000**, far surpassing the level of most human players. This project was recognized by [Dong Nguyen](https://www.linkedin.com/in/dong-nguyen-095714138/), the creator of Flappy Bird.

<p align="center">
  <img src="https://user-images.githubusercontent.com/46636857/77217879-87ed6e80-6b58-11ea-9110-a8c605c190b2.gif">
</p>

## Table of Contents

- [Overview](#overview)
- [Implemented Algorithms](#implemented-algorithms)
- [Environment Design](#environment-design)
- [State Representations](#state-representations)
- [Reward Shaping](#reward-shaping)
- [Network Architectures](#network-architectures)
- [Training Methodology](#training-methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Authors](#authors)

## Overview

Flappy Bird presents a deceptively challenging reinforcement learning problem: the agent must learn a precise control policy in a continuous-state, discrete-action environment with sparse rewards and high penalty for failure. A single mistimed action results in immediate termination, demanding that the learned policy exhibit near-zero error rates over thousands of sequential decisions.

This project implements a **progressive hierarchy of RL algorithms**, enabling systematic analysis of how different value function representations — from tabular methods to deep neural networks — affect sample efficiency, convergence behavior, and asymptotic performance:

| # | Agent | Value Representation | State Space | Key Technique |
|---|-------|---------------------|-------------|---------------|
| 1 | **Baseline** | None (random policy) | N/A | Stochastic action selection |
| 2 | **SARSA** | Tabular Q-table | Discretized 3D | On-policy TD(0) |
| 3 | **Q-Learning** | Tabular Q-table | Discretized 3D | Off-policy TD(0) |
| 4 | **Function Approximation (Linear)** | Weight vector **w** | Continuous features | Linear VFA with SGD |
| 5 | **Deep Q-Network (DNN)** | Feed-forward neural network | Continuous 3D vector | Nonlinear VFA with backpropagation |
| 6 | **Deep Q-Network (CNN)** | Convolutional neural network | Raw 80×80 pixel frames | Experience replay + visual feature extraction |

## Implemented Algorithms

### 1. Baseline Agent (Random Policy)

A non-learning stochastic agent that selects the **flap** action with probability `p` at each timestep. Serves as the performance lower bound to quantify the value added by each learning algorithm.

### 2. SARSA (On-Policy TD Control)

An **on-policy** temporal-difference method that updates Q-values using the action *actually taken* by the current policy, making it inherently more conservative than Q-Learning:

$$Q(s_t, a_t) \leftarrow (1 - \alpha)\, Q(s_t, a_t) + \alpha \left[ r_t + \gamma\, Q(s_{t+1}, a_{t+1}) \right]$$

where $a_{t+1}$ is sampled from the current $\varepsilon$-greedy policy. Because SARSA evaluates the policy it follows, it naturally accounts for exploration noise during updates, leading to safer (more conservative) learned behaviors.

### 3. Q-Learning (Off-Policy TD Control)

An **off-policy** method that decouples the behavior policy from the target policy by bootstrapping on the greedy maximum over next-state Q-values:

$$Q(s_t, a_t) \leftarrow (1 - \alpha)\, Q(s_t, a_t) + \alpha \left[ r_t + \gamma\, \max_{a'} Q(s_{t+1}, a') \right]$$

This allows the agent to learn the optimal policy $Q^*$ regardless of the exploration strategy, typically converging faster than SARSA but with higher variance in stochastic environments.

Both tabular agents support **forward** and **backward** update ordering over episode trajectories. Backward updates propagate terminal-state information more efficiently through the trajectory, improving credit assignment to earlier state-action pairs.

### 4. Function Approximation with Linear Regression

Replaces the Q-table with a **parameterized linear model** $Q(s, a) = \mathbf{w}^\top \boldsymbol{\phi}(s, a)$, where $\boldsymbol{\phi}$ is a hand-crafted feature vector. Weights are updated via semi-gradient descent on the TD error:

$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \left[ Q(s_t, a_t) - \left( r_t + \gamma \max_{a'} Q(s_{t+1}, a') \right) \right] \boldsymbol{\phi}(s_t, a_t)$$

**Feature vector** $\boldsymbol{\phi}(s, a)$:

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `player_vel` | Bird's vertical velocity | ÷ 10 |
| `hor_dist_to_next_pipe` | Horizontal distance to next pipe gap | ÷ 288 |
| `ver_dist_to_next_pipe` | Vertical distance to next pipe gap | ÷ 512 |
| `action` | Action taken (0 or 1) | — |
| `bias` | Constant bias term | 1 |

This enables generalization across similar states, dramatically reducing the effective state space from the tabular setting while maintaining interpretable weights.

### 5. Deep Q-Network (Feed-Forward)

Extends function approximation to a **nonlinear regime** with a multi-layer perceptron that learns features end-to-end via backpropagation:

```
Input [3] → Linear(3, 50) → ReLU → Linear(50, 20) → ReLU → Linear(20, 2) → ReLU → Output [2]
```

The network maps the 3D continuous state vector directly to Q-values for each action, trained with MSE loss between predicted Q-values and TD targets using SGD:

$$\mathcal{L} = \left( Q_\theta(s_t, a_t) - \left[ r_t + \gamma \max_{a'} Q_\theta(s_{t+1}, a') \right] \right)^2$$

### 6. Deep Q-Network (Convolutional)

The most sophisticated agent operates directly on **raw pixel observations**, eliminating the need for hand-crafted state representations. This implements a full DQN pipeline inspired by [Mnih et al. (2015)](https://www.nature.com/articles/nature14236), with key adaptations for the Flappy Bird domain:

```
Input [1×80×80] → Conv2d(1, 16, k=5, s=2) → ReLU
               → Conv2d(16, 32, k=5, s=2) → ReLU
               → Flatten [10,368]
               → Linear(10368, 2) → Output [2]
```

**Key components:**

- **Image preprocessing pipeline**: Raw game frames are cropped (removing the score display), converted to grayscale, resized to 80×80, and binarized via Otsu thresholding at intensity 128 — producing clean, noise-free binary feature maps.
- **Experience replay buffer**: A FIFO buffer of 50,000 transitions is maintained, with uniform random minibatch sampling (batch size 32) to break temporal correlations between consecutive observations — critical for stable convergence in nonlinear function approximation.
- **Warm-up period**: Training begins only after 5,000 transitions have been collected, ensuring sufficient diversity in the replay buffer before gradient updates commence.
- **Survival masking**: Terminal states are handled via a binary survival flag $d_t \in \{0, 1\}$ in the target computation: $y_t = r_t + \gamma \cdot d_t \cdot \max_{a'} Q_\theta(s_{t+1}, a')$, preventing bootstrap from non-existent successor states.
- **Adam optimizer**: Replaces SGD with adaptive learning rates for more robust convergence in the high-dimensional parameter space (~350K parameters).

## Environment Design

The environment wraps the Flappy Bird game from the **PyGame Learning Environment (PLE)** via OpenAI Gym, with custom wrappers that provide algorithm-specific state representations and a unified reward interface.

**Action space**: Binary $\mathcal{A} = \{0, 1\}$ — flap (0) or no-op (1).

Four specialized Gym wrappers (`FlappyBirdNormal`, `FlappyBirdLR`, `FlappyBirdDNN`, `FlappyBirdCNN`) provide distinct observation interfaces while maintaining a consistent reward structure across all agents.

## State Representations

The project explores a progression of state representations, from coarse discrete encodings to raw visual inputs:

| Agent | State Type | Dimensionality | Description |
|-------|-----------|----------------|-------------|
| SARSA / Q-Learning | Discretized string | ~10³ buckets | `(player_vel, hor_dist, ver_dist)` with configurable bucket width |
| Linear FA | Feature dict | 5 features | Normalized continuous features with bias |
| DNN | Continuous vector | 3 | `[player_vel, hor_dist, ver_dist]` — raw continuous values |
| CNN | Binary image tensor | 1×80×80 = 6,400 | Preprocessed, binarized game frame |

**State discretization** for tabular methods: continuous values are mapped to discrete buckets via $\text{discretize}(x, r) = r \cdot \lfloor x/r \rfloor$, where $r$ is a configurable rounding parameter. This controls the bias-variance tradeoff — smaller $r$ preserves more state information but expands the Q-table; larger $r$ enables faster convergence at the cost of state aliasing.

## Reward Shaping

A carefully engineered reward function provides dense learning signal while maintaining alignment with the true game objective:

| Event | Reward | Purpose |
|-------|--------|---------|
| Survive a timestep | **+0.5** | Dense survival signal to encourage longevity |
| Pass through a pipe | **+5.0** | Primary objective reinforcement |
| Collision / death | **−1000** | Strong terminal penalty for constraint violation |

The combination of dense survival reward (+0.5 per step) with sparse milestone reward (+5 per pipe) accelerates early-stage learning while the large death penalty (−1000) ensures the agent strongly prioritizes survival. This asymmetric reward landscape encodes a risk-averse prior that mirrors the game's mechanics — a single mistake ends the episode.

## Network Architectures

### Feed-Forward DNN

| Layer | Input | Output | Activation | Parameters |
|-------|-------|--------|------------|------------|
| Linear 1 | 3 | 50 | ReLU | 200 |
| Linear 2 | 50 | 20 | ReLU | 1,020 |
| Linear 3 | 20 | 2 | ReLU | 42 |
| **Total** | | | | **~1,262** |

### Convolutional DQN

| Layer | Input | Output | Config | Parameters |
|-------|-------|--------|--------|------------|
| Conv2d 1 | 1×80×80 | 16×38×38 | k=5, s=2 | 416 |
| Conv2d 2 | 16×38×38 | 32×17×17 | k=5, s=2 | 12,832 |
| Flatten | 32×17×17 | 9,248 | — | 0 |
| Linear | 9,248 | 2 | — | 18,498 |
| **Total** | | | | **~31,746** |

## Training Methodology

### Exploration Strategy

All agents use an **$\varepsilon$-greedy policy** with a configurable flap probability bias `probFlap` for random action selection. During exploration, the agent flaps with probability `probFlap` (default 0.1) rather than uniform random, reflecting the domain knowledge that the no-flap action is generally safer.

**Epsilon decay** (optional): $\varepsilon_t = \varepsilon_0 / (t + 1)$, transitioning from exploration to exploitation as the policy improves.

**Learning rate decay** (optional): $\alpha_t = \alpha_0 \cdot t / (t + 1)$, reducing step sizes to stabilize convergence.

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Training episodes | 10,000 | Total episodes for training |
| Evaluation frequency | Every 250 episodes | Periodic greedy policy evaluation |
| Evaluation episodes | 1,000 | Episodes per evaluation round |
| Discount factor $\gamma$ | 1.0 | Undiscounted (infinite horizon) |
| Learning rate $\alpha$ | 0.1 – 0.9 | Algorithm-dependent |
| Replay buffer size | 50,000 | CNN agent only |
| Minibatch size | 32 | CNN agent only |
| Warm-up transitions | 5,000 | CNN agent only |

### Update Ordering

Tabular and DNN agents support **forward** and **backward** trajectory updates. Backward updates process transitions from the terminal state first, allowing the strong death penalty signal (−1000) to propagate backward through the episode more efficiently — particularly effective for long episodes where credit assignment is challenging.

## Results

After 10,000 training episodes, the trained agents consistently achieve **scores exceeding 1,400** (pipes passed), with a **peak in-game score of 2,069** — representing superhuman performance in a game that most human players find difficult to score beyond 20–50.

Key findings from the paper:

- **Q-Learning with backward updates** achieves the strongest overall performance, benefiting from off-policy learning and efficient credit assignment.
- **SARSA** exhibits more conservative behavior due to on-policy updates but achieves comparable asymptotic performance.
- **State discretization** (rounding parameter) has a critical impact on tabular methods — too fine leads to sparse Q-tables, too coarse loses discriminative power.
- **Function approximation** methods generalize across the continuous state space, avoiding the curse of dimensionality inherent in tabular approaches.
- **The CNN agent** demonstrates that policies can be learned directly from raw pixels, albeit with higher sample complexity due to the need to learn visual features from scratch.

## Installation

### Dependencies

| Library | Purpose |
|---------|---------|
| [PyTorch](https://pytorch.org) | Neural network training and inference |
| [OpenAI Gym](https://gym.openai.com) | RL environment interface |
| [PyGame Learning Environment](https://github.com/ntasfi/PyGame-Learning-Environment) | Game backend |
| [gym-ple](https://github.com/lusob/gym-ple) | PLE-to-Gym adapter |
| [Pygame](https://www.pygame.org) | Game rendering engine |
| [OpenCV](https://opencv.org) | Image preprocessing for CNN |
| NumPy | Numerical computation |

## Usage

### Training Agents

**Baseline** (random policy):
```bash
python main.py --algo Baseline --probFlap 0.5
```

**SARSA** (on-policy TD with discretization):
```bash
python main.py --algo SARSA --probFlap 0.1 --rounding 10 --lr 0.8
```

**Q-Learning** (off-policy TD with backward updates):
```bash
python main.py --algo QLearning --probFlap 0.1 --rounding 10 --lr 0.8 --order backward
```

**Linear Function Approximation**:
```bash
python main.py --algo FuncApproxLR --probFlap 0.1 --lr 0.1
```

**Deep Q-Network (Feed-Forward)**:
```bash
python main.py --algo FuncApproxDNN --probFlap 0.1 --lr 0.1
```

**Deep Q-Network (Convolutional + Experience Replay)**:
```bash
python main.py --algo FuncApproxCNN --probFlap 0.1 --lr 0.1 --batch-size 32 --num-experience 50000
```

### Full CLI Options

```
--algo              Algorithm: Baseline, QLearning, SARSA, FuncApproxLR, FuncApproxDNN, FuncApproxCNN
--rounding          State discretization bucket width (tabular methods)
--probFlap          Flap probability during ε-greedy exploration (default: 0.1)
--order             Update order: forward | backward (default: forward)
--discount          Discount factor γ (default: 1.0)
--numTrainIters     Training episodes (default: 10,000)
--numTestIters      Evaluation episodes per round (default: 1,000)
--evalPerIters      Evaluation frequency (default: 250)
--epsilon           Initial ε for exploration (default: 0.0)
--lr                Learning rate α (default: 0.1)
--epsilonDecay      Enable ε decay schedule
--lrDecay           Enable learning rate decay schedule
--resume            Resume training from checkpoint (neural network agents)
--seed              Random seed for reproducibility (default: 0)
--batch-size        Minibatch size for experience replay (default: 32)
--num-experience    Replay buffer capacity (default: 50,000)
```

## Project Structure

```
FlapAI-Bird/
├── main.py                          # Entry point with CLI argument parsing
├── agents/
│   ├── TemplateAgent.py             # Abstract base class for all agents
│   ├── BaselineAgent.py             # Random policy baseline
│   ├── SARSAAgent.py                # On-policy SARSA with tabular Q-values
│   ├── QLearningAgent.py            # Off-policy Q-Learning with tabular Q-values
│   ├── FuncApproxLRAgent.py         # Linear regression function approximation
│   ├── FuncApproxDNNAgent.py        # Feed-forward DNN (3→50→20→2)
│   └── FuncApproxCNNAgent.py        # Convolutional DQN with experience replay
├── game/
│   └── FlappyBirdGame.py            # OpenAI Gym wrappers (4 environment variants)
├── utils/
│   └── utils.py                     # Discretization, dot product, vector increment
├── LICENSE
└── README.md
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{vu2020flapai,
  title={FlapAI Bird: Training an Agent to Play Flappy Bird Using Reinforcement Learning Techniques},
  author={Vu, Tai and Tran, Leon},
  journal={arXiv preprint arXiv:2003.09579},
  year={2020}
}
```

## Authors

- **Tai Vu** — Stanford University
- **Leon Tran** — Stanford University
