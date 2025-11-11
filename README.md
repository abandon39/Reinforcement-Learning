# Reinforcement-Learning
# DQN Solutions for MountainCar-v0

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org/)
[![OpenAI Gym](https://img.shields.io/badge/Environment-OpenAI%20Gym-green)](https://gym.openai.com/)

This repository documents my implementation of DQN-based solutions for the **MountainCar-v0** environment.

## 📁 Project Structure

### Core Implementations
- **`DQN_model/`** - Contains implementations of:
  - Standard DQN
  - Double DQN  
  - Dueling DQN
  - PER DQN (Prioritized Experience Replay)
  - Rainbow DQN (simplified version)
  - My custom modified DQN variant

### Supporting Modules
- **`Network/`** - Neural network architectures used by these algorithms
- **`Replay_buffer/`** - Experience replay buffer designs (both standard and prioritized)

### Key Files
- **`changed_train_test_visualize.py`** - Training and visualization script for my modified DQN version
- **`train_visualize.py`** - Training script for all other standard DQN variants

### Results & Visualizations
- Various **PNG files** - Result visualizations and analysis plots
- **GIF files** - Animations of successful runs, including the **83-step ascent** mentioned in the course practice report
