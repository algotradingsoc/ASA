from collections import deque
import numpy as np
import os
import random

import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, CuDNNLSTM
from keras.layers.merge import Add, Multiply, Concatenate
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf

class ActorCritic:
    def __init__(self, constants, sess):
        self.constants = constants

        self.variables = {'state':None,
                          'next_state': None,
                          'action': None,
                          'reward': 0,
                          'done': False, 
                          'rnd_choice': 1}
        
        self.sess = sess #tensorflow session
        self.memory = deque(maxlen=self.constants['memory'])
        self.order_memory = deque(maxlen=self.constants['order_memory'])

        ##hyper-parameters
        self.batch_size = 32
        self.gamma = 0.99
        self.tau = 0.125
        if self.constants['train']:
            self.order_epsilon = 1.0
            self.empty_epsilon = 1.0
        else:
            self.order_epsilon = 0.001
            self.empty_epsilon = 0.001
        self.order_epsilon_min = 0.001
        self.empty_epsilon_min = 0.01
        self.order_epsilon_decay = 0.9
        self.empty_epsilon_decay = 0.99
        self.learning_rate = 0.00001

        ##actor model
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        #load weights if 'load_model' == True
        if self.constants['load_model']:
            name = "target_actor"
            self.model = self.load(f"models/{name}_weights.h5", self.model)

        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.constants['action_size']]) #feed de/dC (from critic)

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, -self.actor_critic_grad) #compute dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        ##critic model
        self.critic_state_input, self.critic_action_input, \
			self.critic_model = self.create_critic_model()
        _,_, self.target_critic_model = self.create_critic_model()

        #load weights if 'load_model' == True
        if self.constants['load_model']:
            name = "target_critic"
            self.model = self.load(f"models/{name}_weights.h5", self.model)

        self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input) #compute de/dC

        self.sess.run(tf.global_variables_initializer())

    def create_actor_model(self):
        #MLP model
        state_input = Input(shape=(self.constants['inst_state_size'],))
        h1 = Dense(16, activation='relu')(state_input)
        h2 = Dense(16, activation='relu')(h1)
        h3 = Dense(16, activation='relu')(h2)
        output = Dense(self.constants['action_size'], activation='linear')(h3)

        ##RNN
        ma_diff_inputs = Input(shape=(self.constants['ma_diff_buffer_size'],1),
                               name='ma_diff_input')
        ma_diff_x = CuDNNLSTM(16, return_sequences=True,
                              name='lstm_after_inputs')(ma_diff_inputs)
        ma_diff_x = CuDNNLSTM(16, name='lstm_before_merge')(ma_diff_x)

        """ Merges RNN wih MLP """
        merge_x = Concatenate()([output, ma_diff_x])
        merge_x = Dense(16, activation='relu')(merge_x)
        merge_output = Dense(self.constants['action_size'], 
                             activation='linear')(merge_x)
    
        model = Model([state_input, ma_diff_inputs], merge_output)
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=(self.constants['inst_state_size'],))
        action_input = Input(shape=(self.constants['action_size'],))
        
        s1 = Dense(32, activation='relu')(state_input)
        s2 = Dense(32, activation='relu')(s1)
        s3 = Dense(16, activation='linear')(s2)
        
        a1 = Dense(32, activation='relu')(action_input)
        a2 = Dense(32, activation='relu')(a1)
        a3 = Dense(16, activation='linear')(a2)

        merged = Add()([s3, a3])
        x = Dense(16, activation='relu')(merged)
        output = Dense(1, activation='linear')(x)
        model = Model(input=[state_input, action_input], output=output)

        model.compile(loss="mse", 
                      optimizer=Adam(lr=self.learning_rate))
        return state_input, action_input, model

    ##model training

    def main_loop(self, memory, 
                  inst_inputs, lstm_inputs, orders, 
                  reward=0, done=False, new_action=True):
        self.variables['reward'] = reward
        self.variables['done'] = done
        self.update_state(inst_inputs, lstm_inputs)
        if self.variables['state'] is None:
            self.variables['state'] = self.variables['next_state']
            return memory
        if new_action:
            self.variables['action'], self.variables['rnd_choice'] = self.get_action(self.variables['state'], orders)
        memory = self.remember(memory, 
                               self.variables['state'], self.variables['action'],
                               self.variables['reward'], self.variables['next_state'],
                               self.variables['done'], self.variables['rnd_choice'])
        self.variables['state'] = self.variables['next_state'] 
        return memory


    def remember(self, memory, state, action, reward, next_state, done, rnd_choice):
        memory.append((state, action, reward, next_state, done, rnd_choice))
        return memory

    def _train_actor(self, samples):

        for state, _, _, _, _ in samples:

            predicted_action = self.actor_model.predict(state[0], state[1])
            grads = self.sess.run(self.critic_grads, feed_dict = {self.critic_state_input: state, self.critic_action_input: predicted_action})[0]

            self.sess.run(self.optimize, feed_dict = {self.actor_state_input: state, self.actor_critic_grad: grads})

    def _train_critic(self, samples):
        
        print(np.asarray(samples).shape)
        for state, action, reward, next_state, done in samples:

            if not done:
                target_action = self.target_actor_model.predict(next_state)
                future_reward = self.target_critic_model.predict([next_state, target_action])[0][0]
                reward += self.gamma * future_reward
            
            self.critic_model.fit([state, action], reward)

    def train(self):

        if len(self.memory) < self.batch_size:
            return

        random.seed(2)

        samples = random.sample(self.memory, self.batch_size)
        print(len(samples))
        self._train_critic(samples)
        self._train_actor(samples)

    ##target model update

    def _update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = self.tau * actor_model_weights[i] + (1 - self.tau)* actor_target_weights[i]
        self.target_actor_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = self.tau * critic_model_weights[i] + (1 - self.tau)* critic_target_weights[i]
        self.target_critic_model.set_weights(critic_target_weights)

    def _update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    def get_action(self, state, orders):

        if orders:
            self.order_epsilon *= self.order_epsilon_decay
            if np.random.rand() <= self.order_epsilon:
                return self.make_random_choice()
        else:
            self.empty_epsilon *= self.empty_epsilon_decay
            if np.random.rand() <= self.empty_epsilon:
                return self.make_random_choice()

        act_values = self.actor_model.predict([state[0],
                                               state[1]])

        if self.constants['debug']:
            print(np.argmax(act_values[0]), "+")

        return np.argmax(act_values[0]), 0

    def update_state(self, inst_inputs, lstm_inputs):
        """ 
        Returns list of inst inputs and lstm inputs
        inst_inputs: 
         - self.bid_diff       : Change in Bid from last previous
         - self.ask_diff       : Change in Ask from last previous
         - self.spread         : Difference between ask and diff
         - self.order_dir      : Order type (0 = no order, 1 = buy order, 
                                             -1 = sell order)
         - self.max_drawdown   : Difference since the lowest 
                                 point in current order
         - self.max_upside     : Difference since the highest point 
                                 in current order
        lstm_inputs:
         - self.ma_diff_buffer : Array of difference in price at 
                                 intervals of diff_step 
        """
        lstm = np.array([lstm_inputs])
        inst = np.reshape(inst_inputs, [1, len(inst_inputs)])
        lstm = np.reshape(lstm, (1, lstm.shape[1], 1))
        self.variables['next_state'] = [inst, lstm]
        return

    def make_random_choice(self):
        rnd_choice = random.randrange(self.constants['action_size'])
        if self.constants['debug']:
            print(rnd_choice, "-")
        return rnd_choice, 1  ## Random choice was made
        
    def load(self, name, model):
        """ Loads the weights into the Neural Network """
        if os.path.isfile(name):
            print(f'Loaded {name}')
            model.load_weights(name)
        return model
            
    def save(self, name, model):
        """ Saves a local copy of the model weights """
        model.save_weights(name)






