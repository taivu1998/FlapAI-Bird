# FlapAI-Bird

This AI program implements several AI agents for playing Flappy Bird. The program applies reinforcement learning techniques, including SARSA, Q-Learning, and Function Approximation.

## Installation

The project requires the following frameworks:

- Pygame: https://www.pygame.org

- PyGame Learning Environment: https://github.com/ntasfi/PyGame-Learning-Environment

- OpenAI Gym: https://gym.openai.com

- gym-ple: https://github.com/lusob/gym-ple
 
- PyTorch: https://pytorch.org

- OpenCV: https://opencv.org

## Usage

The program implements the following agents.

- Baseline Agent.

```bash
python main.py --algo=Baseline --probFlap=0.5
```

- SARSA Agent.

```bash
python main.py --algo=SARSA --probFlap=0.1 --rounding=10 --lr=0.8
```

- Q-Learning Agent.

```bash
python main.py --algo=QLearning --probFlap=0.1 --rounding=10 --lr=0.8 --order=backward
```

- Function Approximation Agent with Linear Regression.

```bash
python main.py --algo=FuncApproxLR --probFlap=0.1 --lr=0.1
```

- Function Approximation Agent with a Feed Forward Neural Network.

```bash
python main.py --algo=FuncApproxDNN --probFlap=0.1 --lr=0.1
```

- Function Approximation Agent with Convolutional Neural Network.

```bash
python main.py --algo=FuncApproxCNN --probFlap=0.1 --lr=0.1
```

## Authors

* **Tai Vu** - Stanford University

* **Leon Tran** - Stanford University
