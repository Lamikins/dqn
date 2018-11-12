# Deep Q-Network and Double Deep Q-Network for Game Playing in Tensorflow

This repository is an implementation of the DQN and Double-DQN from DeepMind

The Deep Q-Network was a seminal deep reinforcement learning paper by Mnih et al., in which the authors demonstrate a system for learning control policies for Atari games directly from visual input.  

### Usage

Clone the repository, then install all necessary required packages.  Training can be started on the OpenAI Gym Cart Pole environment simply by running:

```
python learn.py
```

This instantiates a Double Deep Q-Net and trains for 1000 episodes with parameters {image_size: 28x28, learning_rate: 1e-3, sample_time: 1000}

You can modify the hyperparameters in learn.py and change the agent to be either a DoubleDQNLearner or DQNLearner.  