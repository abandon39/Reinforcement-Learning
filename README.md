# Reinforcement-Learning
This repository documents my implementation of DQN-based solutions for the MountainCar-v0 environment. The project includes the following components:

The DQN_model folder contains implementations of:
  Standard DQN
  Double DQN
  Dueling DQN
  PER DQN (Prioritized Experience Replay)
  Rainbow DQN (simplified version)
  My custom modified DQN variant

The Network folder contains neural network architectures used by these algorithms
The Replay_buffer folder houses experience replay buffer designs, including both standard and prioritized replay implementations

Key files:
  changed_train_test_visualize.py: Training and visualization script for my modified DQN version
  train_visualize.py: Training script for all other standard DQN variants

The repository also includes various PNG and GIF visualization files. Notably, the GIF file demonstrates a successful ascent to the summit in just 83 steps, as referenced in the course practice report.
