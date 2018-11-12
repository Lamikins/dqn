import numpy as np
from tqdm import tqdm
from scipy import misc
from collections import deque, namedtuple
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from utils import linear_scale, preprocess, LearnerState

class DQNLearner(object):
    #Args 
    # imsize: int of side of image length/width
    # num_actions: int; the number of actions the agent can take
    # discount: float; the discount factor to multiply by
    # sample_time: int; number of steps to run the game before training starts (to populate memory)
    # lr: float; learning rate
    # batch_size: int of batch size
    # max_memory: maximum number of trials to store in memory
    # eps: a float between [0,1] indicating the starting epsilon value for eps-greedy
    # num_eps: number of steps to get from "eps" to 0.
    
    def __init__(self,num_actions,
                 imsize=32,
                 discount=.999,
                 sample_time=4000,
                 lr=4e-5,
                 batch_size=128,
                 max_memory=50000,
                 eps=1.0,
                 num_eps = 10000,
                 logdir = "dqn_results/"):
        
        self.num_actions = num_actions
        self.imsize = imsize
        self.discount = discount
        self.sample_time = sample_time
        self.lr = lr
        self.batch_size = batch_size
        self.max_memory = max_memory
        
        #logdir for saving tensorboard outputs
        self.logdir=logdir
        
        #this is the number of frames until we reach the final eps
        self.num_eps = num_eps
        #starting eps
        self.eps = eps
        
        self.sess = None
        self.time = 0
        self.init_time = 0
        self.episode_time = 0
        self.episode_rewards = []
        self.average_q = []
        self.remember_length = 4
        
        #for non-conv
        self.state_size = 4
        
        #eps annealing
        self.final_eps = 0
        self.start_eps = self.eps
        
        self.losses = []
        self.rewards = []
        
        #initialize image memory and other memory
        self.image_memory = deque([np.zeros((self.imsize, self.imsize,1)) for i in range(self.remember_length)],
                            maxlen=self.remember_length)
        self.memory = deque(maxlen=self.max_memory)

    #memory should be filepath+'.npy' and checkpoint should be filepath+".ckpt"
    def load_from_path(self,filepath):
        self.train_saver.restore(self.sess,filepath+".ckpt")
        self.memory = deque(np.load(filepath+'.npy'))
        
    def save_to_path(self,filepath):
        self.train_saver.save(self.sess,os.getcwd()+"/models/%s.ckpt" % filepath)
        np.save(os.getcwd() + "/models/%s.npy" % filepath,self.memory)
        
    def initialize_network(self):
        self.create_q_network()
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)
    
    def create_q_network(self):
        self.g = tf.Graph()
        
        def count_parameters():
            num_params = 0
            for var in tf.trainable_variables():
                num_params += int(np.prod(var.shape))
            return num_params
        
        with self.g.as_default():
            #We use the last four frames
            X = tf.placeholder(tf.float32,shape=(None,self.imsize,self.imsize,self.remember_length),name="input")
            y = tf.placeholder(tf.float32,shape=(None,self.num_actions),name="labels")

            filters = 8
            act = tf.nn.relu
            
            self.activations = []
            initializer = tf.truncated_normal_initializer(0.0,1e-2)
            
            net = tf.layers.Conv2D(filters,4,2,padding="same",activation=act, kernel_initializer=initializer)(X)
            self.activations.append(net)
            
            net = tf.layers.Conv2D(filters*2,4,2,padding="same",activation=act,kernel_initializer=initializer)(net)
            self.activations.append(net)
            
            net = tf.layers.Conv2D(filters*2,3,1,padding="same",activation=act,kernel_initializer=initializer)(net)
            self.activations.append(net)
            
            net = tf.layers.flatten(net)
            self.activations.append(net)
            
            net = tf.layers.Dense(256,activation=act,kernel_initializer=initializer)(net)
            self.activations.append(net)
            print("Initialized network with %d parameters" % count_parameters())
        
            logits = tf.layers.Dense(self.num_actions,activation=None,
                                     kernel_initializer=initializer,name="output")(net)
            loss = tf.reduce_mean(self.huber_loss(logits-y))
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
            gradients = tf.gradients(loss, tf.trainable_variables())
            gradients
            gradients, _ = tf.clip_by_global_norm(gradients,10)
            train_op = optimizer.minimize(loss)

            now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            logdir = "%s/run-%s/" % (self.logdir,now)

            #Getting the histogram summaries
            hists = [tf.summary.histogram(var.name, var) for var in self.activations] +\
                [tf.summary.histogram(var.name, var) for var in tf.trainable_variables()] +\
                [tf.summary.histogram(var.name, var) for var in gradients]

            metric_summaries = [tf.summary.scalar("Loss", loss)]
            self.tf_summary = tf.summary.merge(metric_summaries)
            self.tf_hist_summary = tf.summary.merge(hists)
            self.train_writer = tf.summary.FileWriter(logdir,graph =self.g)

            self.train_saver = tf.train.Saver(tf.trainable_variables())

            self.train_op = train_op
            self.loss = loss
            self.logits = logits
            self.init = tf.global_variables_initializer()

    def get_memory_batch(self):
        if len(self.memory) < self.batch_size:
            inds = np.random.choice(np.arange(len(self.memory)),replace=True,size=self.batch_size)
        else:
            inds = np.random.choice(np.arange(len(self.memory)),replace=False,size=self.batch_size)
        return [self.memory[i] for i in inds]
        
    
    def huber_loss(self, x, delta=1.0):
        """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
        return tf.where(
            tf.abs(x) < delta,
            tf.square(x) * 0.5,
            delta * (tf.abs(x) - 0.5 * delta)
        )

    def reset_memory(self):
        self.image_memory = deque([np.zeros((self.imsize, self.imsize,1)) for i in range(self.remember_length)],
                            maxlen=self.remember_length)
        
    def reset_learner_state(self):
        self.last_learner_state = LearnerState(np.zeros((self.imsize,self.imsize,self.remember_length)),0)
    
    def eps_greedy(self,eps,q_values_of_state):
        if np.random.choice([True,False],p=[eps,1-eps]):
            action = np.random.choice(np.arange(self.num_actions))
        else:
            action = np.argmax(q_values_of_state)

        return action
    
    def action_callback(self, state):
        #Sample if necessary (taking random actions)
        if (self.init_time < self.sample_time):
            current_screen = preprocess(state['pixels'],self.imsize)
            self.image_memory.append(current_screen)
            current_state = np.concatenate(self.image_memory,axis=2)
            last_action = np.random.choice(np.arange(self.num_actions))
            
            #Have to make sure that the last reward state is not zero. This
            #will provide the 'basis' for our network to learn faster.  Helps initialize
            #to stable q-values.
            #if not np.all(np.mean(current_state,(0,1)) == 0) and self.last_learner_state.last_reward != 0:
            if self.last_learner_state.last_reward != 0:
                self.memory.append((self.last_learner_state.last_state,
                                   current_state,
                                   last_action,
                                   self.last_learner_state.last_reward))
            self.init_time +=1
                
            if self.init_time == self.sample_time:
                print("Stored %d initial memories" % len(self.memory))
            self.last_learner_state = LearnerState(current_state,None)
            return last_action
        else:
            #First we need to get current state.
            current_screen = preprocess(state['pixels'],self.imsize)
            #plt.imshow(current_screen[:,:,0])
            #plt.show()
            self.image_memory.append(current_screen)
            
            #This returns a (32,32,4) state.
            current_state = np.concatenate(self.image_memory,axis=2)

            #Get q-values of last state and get last action
            last_q_values = self.sess.run(self.logits,feed_dict={"input:0":[self.last_learner_state.last_state]})
            
            self.average_q.append(np.mean(last_q_values))
            
            last_action = self.eps_greedy(self.eps, last_q_values)

            #append to memory
            self.memory.append((self.last_learner_state.last_state,
                               current_state,
                               last_action,
                               self.last_learner_state.last_reward))
            
            #set the last learner state
            self.last_learner_state = LearnerState(current_state,None)

            #now we perform learning
            memory_batch = self.get_memory_batch()

            
            #x_batch = np.zeros((self.batch_size,self.imsize,self.imsize,4))
            x_batch = np.concatenate([np.expand_dims(memory_batch[i][0],0) for i in range(len(memory_batch))])
            #y_batch = np.zeros((self.batch_size, self.num_actions))

            css = np.concatenate([np.expand_dims(memory_batch[i][1],0) for i in range(len(memory_batch))])
            yb_in = np.concatenate([np.expand_dims(memory_batch[i][0],0) for i in range(len(memory_batch))])
            a_inds = np.array([memory_batch[i][2] for i in range(len(memory_batch))])
            #need to turn to float, otherwise will be integers!
            target = np.array([memory_batch[i][3] for i in range(len(memory_batch))]).astype(np.float32)

            target_q = self.sess.run(self.logits,feed_dict={'input:0':css})
            lr_g0 = target >= 0
            target[lr_g0] = target[lr_g0] + self.discount * np.amax(target_q[lr_g0],1)

            yb = self.sess.run(self.logits, feed_dict={"input:0":yb_in})
            yb[np.arange(a_inds.shape[0]),a_inds] = target
            y_batch = yb
            
            #x_batch = (x_batch - np.mean(x_batch)) / (np.std(x_batch)+1e-6)
            loss, sstr, hstr, _ = self.sess.run([self.loss, self.tf_summary,
                                     self.tf_hist_summary, self.train_op],feed_dict={"input:0":x_batch,
                                                                    "labels:0":y_batch})

            self.losses.append(loss)
            self.time +=1

            #action summary
            action_summary = tf.Summary(value=[tf.Summary.Value(tag='action',simple_value=last_action)])
            self.train_writer.add_summary(action_summary,self.time)
            self.last_learner_state.last_action = last_action
            
            #eps summary
            self.eps -= (self.start_eps - self.final_eps) / self.num_eps
            self.eps = max(self.final_eps,self.eps)
            eps_summary = tf.Summary(value=[tf.Summary.Value(tag="eps", simple_value=self.eps)])
            
            if self.time % 10 == 0:
                self.train_writer.add_summary(sstr,self.time)
                self.train_writer.add_summary(eps_summary,self.time)
            
            if self.time % 100 == 0:
                self.train_writer.add_summary(hstr,self.time)
            
            return last_action

    def reward_callback(self, reward):
        #Reward of < 0 is end state
        if reward < 0:
            self.episode_time +=1
            mean_rewards = np.sum(self.episode_rewards)
            
            #average q summary
            avg_q_summary = tf.Summary(value=[tf.Summary.Value(tag='average_q',simple_value=np.mean(self.average_q))])
            
            self.average_q = []
            
            #not really an average, right now!
            rewards_summary = tf.Summary(value=[
                    tf.Summary.Value(tag="average_episode_reward", simple_value=mean_rewards), 
                ])
            
            self.train_writer.add_summary(avg_q_summary,self.episode_time)
            self.train_writer.add_summary(rewards_summary,self.episode_time)
            
            self.episode_rewards = []
                   
        self.episode_rewards.append(reward)
        self.rewards.append(reward)
        #reward summary
        reward_summary = tf.Summary(value=[tf.Summary.Value(tag="reward", simple_value=reward)])
        self.train_writer.add_summary(reward_summary,self.time)
        self.last_learner_state.last_reward = reward

    def close(self):
        self.sess.close()
        

class EvalLearner(object):
    def __init__(self, filepath,imsize=32):
        self.sess =None
        self.imsize = imsize
        self.filepath = filepath
        self.remember_length = 4
        self.image_memory = deque([np.zeros((self.imsize, self.imsize,1)) for i in range(self.remember_length)],
                            maxlen=self.remember_length)

    def initialize_network(self):
        self.train_saver = tf.train.import_meta_graph(self.filepath+".meta")
        self.sess = tf.Session()
        self.train_saver.restore(self.sess,self.filepath)

    def reset_learner_state(self):
        self.last_learner_state = LearnerState(np.zeros((self.imsize,self.imsize,self.remember_length)),0)
    
    def action_callback(self, state):
        #First we need to get current state.
        current_screen = preprocess(state['pixels'],self.imsize)
        self.image_memory.append(current_screen)
        #This returns a (32,32,4) state.
        current_state = np.concatenate(self.image_memory,axis=2)
        #Get q-values of last state and get last action
        last_q_values = self.sess.run("output:0",feed_dict={"input:0":[self.last_learner_state.last_state]})
        last_action = np.argmax(last_q_values)
        #set the last learner state
        self.last_learner_state = LearnerState(current_state,None)
        return last_action

    def reward_callback(self, reward):
        self.last_learner_state.last_reward = reward

    def close(self):
        self.sess.close()


