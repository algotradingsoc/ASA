from collections import deque
import numpy as np
import os
import random

import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, LSTM
from keras.optimizers import Adam
from keras import backend as K


class DeepQNN:
    def __init__(self, train, agent_name, constants):
        self.train = train
        self.agent_name = agent_name
        self.constants = constants
        
        self.action_size = 4 ## buy, sell, cancel, do nothing
        self.batch_size = 4
        
        self.memory = deque(maxlen=self.constants['memory'])
        self.order_memory = deque(maxlen=self.constants['order_memory'])
        self.gamma = 0.9
        if self.train:
            self.order_epsilon = 1.0
            self.empty_epsilon = 1.0
        else:
            self.order_epsilon = 0.001
            self.empty_epsilon = 0.001
        self.order_epsilon_min = 0.001
        self.empty_epsilon_min = 0.01
        self.order_epsilon_decay = 0.9
        self.empty_epsilon_decay = 0.99
        self.learning_rate = 0.0001
        
        self.model = self._build_model()
        self.model = self.load(f'models/{self.agent_name}_weights.h5', self.model)
    
    
    
    def remember(self, memory, state, action, reward, next_state, done):
        memory.append((state, action, reward, next_state, done))
        return memory
    
    
    
    def replay(self, memory, batch_size, model, decay=True):
        """
        Trains the model
         - Iterates through a random sample of the memory of size batch_size
        """
        if len(memory) < batch_size:
            return model
        minibatch = random.sample(memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                ## If training step is not complete then tries to predict reward
                prediction = model.predict([next_state[0], next_state[1]])
                target = reward + (self.gamma 
                                   * np.amax(prediction[0])) 
            target_f = model.predict([state[0], state[1]])
            target_f[0][action] = target
            model.fit([state[0],state[1]], 
                      target_f, epochs=1, verbose=0) ## Training action
        if decay:
            ## Decays epsilon values at an exponential rate
            if self.order_epsilon > self.order_epsilon_min:
                self.order_epsilon *= self.order_epsilon_decay
            if self.empty_epsilon > self.empty_epsilon_min:
                self.empty_epsilon *= self.empty_epsilon_decay
        return model
            
    
    
    def get_action(self, state, orders):
        """ 
        Determines the action to take, returns action ID.
         - Depends on the state and the epsilon value
         - Makes random choice with probability of epsilon.
           - self.order_epsilon : probability when the model 
                                  is currently in an order
           - self.empty_epsilon : probability when the model 
                                  is not in an order
        """
        if orders:
            if np.random.rand() <= self.order_epsilon:
                return self.make_random_choice()
        else:
            if np.random.rand() <= self.empty_epsilon:
                return self.make_random_choice()
        ## If not random choice tries to predict best action
        act_values = self.model.predict([state[0], 
                                         state[1]])
        if self.constants['debug']:
            print(np.argmax(act_values[0]), "+")
        return np.argmax(act_values[0]), 0 ## Random choice was not made
    
    
    
    def make_random_choice(self):
        rnd_choice = random.randrange(self.action_size)
        if self.constants['debug']:
            print(rnd_choice, "-")
        return rnd_choice, 1  ## Random choice was made
    
    
    
    def _build_inst_model(self):
        """ Initialiser for the MLP part of the model """
        inst_model = Sequential()
        inst_model.add(Dense(24, 
                             input_dim=self.constants['inst_state_size'], 
                             activation='relu'))
        inst_model.add(Dropout(0.1))
        inst_model.add(Dense(48, activation='relu'))
        inst_model.add(Dropout(0.1))
        inst_model.add(Dense(24, activation='relu'))
        inst_model.add(Dropout(0.1))
        inst_model.add(Dense(self.action_size, activation='linear'))
        inst_model = self.load(f'models/inst_{self.agent_name}_weights.h5', inst_model) 
        return inst_model
        
        
        
    def _build_model(self):
        """ Builds the complete neural network """
        inst = self._build_inst_model()

        """ RNN model """
        ma_diff_inputs = Input(shape=(self.constants['ma_diff_buffer_size'], 1),
                               name='ma_diff_input')
        ma_diff_x = LSTM(32, activation='relu', 
                         return_sequences=True, name='lstm_after_inputs')(ma_diff_inputs)
        ma_diff_x = LSTM(8, activation='relu', 
                         return_sequences=True, name='lstm_mid')(ma_diff_inputs)
        ma_diff_x = LSTM(3, activation='relu', name='lstm_before_merge')(ma_diff_x)

        """ Merges RNN wih MLP """
        merge_x = keras.layers.concatenate([inst.output, ma_diff_x])
        merge_x = Dense(32, activation='relu')(merge_x)
        merge_x = Dropout(0.1)(merge_x)
        merge_output = Dense(self.action_size, activation='linear')(merge_x)

        model = Model([inst.input, ma_diff_inputs], merge_output)
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
            
         
            
    def load(self, name, model):
        """ Loads the weights into the Neural Network """
        if os.path.isfile(name):
            print(f'Loaded {name}')
            model.load_weights(name)
        return model
        
            
            
    def save(self, name, model):
        """ Saves a local copy of the model weights """
        model.save_weights(name)
        
    