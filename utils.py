import numpy as np

def linear_scale(x,newmin,newmax):
    oldmin = np.min(x)
    oldmax = np.max(x)
    return ((x-oldmin) * (newmax-newmin)/(oldmax-oldmin)) + newmin

def preprocess(img,imsize):
    resize = misc.imresize(img,(imsize,imsize),mode='L') / 255.
    return np.expand_dims(resize,2)
    
class LearnerState:
    
    def __init__(self,last_state,last_reward):
        self.last_state = last_state
        self.last_reward = last_reward
        self.last_action = 0
