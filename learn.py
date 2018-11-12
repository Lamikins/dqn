import numpy as np
from tqdm import tqdm 
from agent import DQNLearner
from agent_doubledqn import DoubleDQNLearner
import gym
import matplotlib.pyplot as plt
import tensorflow as tf

def run_games(learner, env, iters = 100):
    learner.initialize_network()
    learner.reset_memory()
    learner.reset_learner_state()
    env.reset()
    
    for ii in tqdm(range(iters)):
        # Reset the state of the learner.
        env.reset()
        done = False
        state = {}
        learner.reset_memory()
        
        while not done:
            
            pixels = env.render(mode="rgb_array")
            env.render()
            pixels = np.mean(pixels,2)
            
            state['pixels'] = pixels
            
            action = learner.action_callback(state)
            
            next_state, reward, done, info = env.step(action)
            
            if done:
                reward = -1
                
            learner.reward_callback(reward)
        
    return

env = gym.make("CartPole-v0")

agent = DoubleDQNLearner(env.action_space.n,sample_time=1000,imsize=28,lr=1e-3,num_eps=1000)
run_games(agent,env,1000)
env.close()