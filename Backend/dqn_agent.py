# backend/dqn_agent.py

import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, Lambda, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras import mixed_precision

# Set mixed precision policy
mixed_precision.set_global_policy('mixed_float16')

@register_keras_serializable()
def combine_streams(inputs):
    """
    Combines the value and advantage streams for Dueling DQN.
    """
    value, advantage = inputs
    advantage_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)
    return value + (advantage - advantage_mean)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.alpha = alpha
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        probs = (priorities + 1e-5) ** self.alpha  # Add small constant to prevent zero probabilities
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_shape, action_size, model=None):
        self.state_shape = state_shape  # e.g., (4, 4, 1)
        self.action_size = action_size  # 4 actions
        self.memory = PrioritizedReplayBuffer(capacity=100000)
        self.gamma = 0.99    # Discount factor
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.05  # Lowered minimum epsilon
        self.learning_rate = 1e-4
        self.epsilon_decay = 0.995  # Faster decay to encourage exploitation
        self.batch_size = 128
        self.beta = 0.4  # Initial value of beta for importance-sampling
        self.beta_increment = 1e-6  # Increment of beta per step
        self.tau = 0.001  # Soft update parameter

        if model is not None:
            self.model = model
            self.epsilon = self.epsilon_min  # Optionally set epsilon to min if resuming
        else:
            self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()  # Initialize target model

        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.loss_function = tf.keras.losses.Huber()

        self.current_loss = 0  # Initialize current loss

    def _build_model(self):
        input_layer = Input(shape=(4, 4, 16))  # Adjusted for one-hot encoded input
        # Convolutional layers with custom kernel sizes
        conv_a = Conv2D(512, kernel_size=(2, 1), activation='relu', padding='valid')(input_layer)
        conv_b = Conv2D(512, kernel_size=(1, 2), activation='relu', padding='valid')(input_layer)

        conv_aa = Conv2D(1024, kernel_size=(2, 1), activation='relu', padding='valid')(conv_a)
        conv_ab = Conv2D(1024, kernel_size=(1, 2), activation='relu', padding='valid')(conv_a)
        conv_ba = Conv2D(1024, kernel_size=(2, 1), activation='relu', padding='valid')(conv_b)
        conv_bb = Conv2D(1024, kernel_size=(1, 2), activation='relu', padding='valid')(conv_b)

        # Flatten and concatenate
        flat_layers = [Flatten()(layer) for layer in [conv_aa, conv_ab, conv_ba, conv_bb, conv_a, conv_b]]
        concatenated = tf.keras.layers.Concatenate()(flat_layers)

        # Fully connected layers
        dense = Dense(256, activation='relu')(concatenated)
        dense = Dense(128, activation='relu')(dense)

        # Output layer
        output = Dense(self.action_size, activation='linear')(dense)

        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """
        Performs a soft update of the target network's weights.
        """
        main_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        new_weights = []
        for mw, tw in zip(main_weights, target_weights):
            new_weights.append(self.tau * mw + (1 - self.tau) * tw)
        self.target_model.set_weights(new_weights)

    def remember(self, state, action, reward, next_state, done):
        # Store experiences with initial priority
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # Use predict_on_batch to suppress verbose output
        act_values = self.model.predict_on_batch(state[np.newaxis, :])
        return np.argmax(act_values[0])  # Returns action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        samples, indices, weights = self.memory.sample(self.batch_size, self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Convert samples to tensors with dtype=float16 to match mixed precision
        states = tf.convert_to_tensor([sample[0] for sample in samples], dtype=tf.float16)
        actions = tf.convert_to_tensor([sample[1] for sample in samples], dtype=tf.int32)
        rewards = tf.convert_to_tensor([sample[2] for sample in samples], dtype=tf.float16)
        next_states = tf.convert_to_tensor([sample[3] for sample in samples], dtype=tf.float16)
        dones = tf.convert_to_tensor([sample[4] for sample in samples], dtype=tf.float16)

        # Cast gamma to float16
        gamma = tf.constant(self.gamma, dtype=tf.float16)

        with tf.GradientTape() as tape:
            # Current Q values
            q_values = self.model(states, training=True)

            # Next Q values from main network for Double DQN
            next_q_main = self.model(next_states, training=False)

            # Next Q values from target network
            next_q_target = self.target_model(next_states, training=False)

            # Double DQN: select actions using main network, evaluate with target network
            a_prime = tf.argmax(next_q_main, axis=1)

            # Ensure tf.one_hot outputs float16 to match next_q_target
            actions_one_hot = tf.one_hot(a_prime, self.action_size, dtype=tf.float16)

            # Compute target Q values
            target_q = rewards + gamma * tf.reduce_sum(next_q_target * actions_one_hot, axis=1) * (1 - dones)

            # Gather Q-values for the taken actions
            actions_one_hot = tf.one_hot(actions, self.action_size, dtype=tf.float16)
            q_pred_actions = tf.reduce_sum(q_values * actions_one_hot, axis=1)

            # Calculate Huber loss
            loss = self.loss_function(target_q, q_pred_actions, sample_weight=weights)

        # Compute gradients and apply
        grads = tape.gradient(loss, self.model.trainable_variables)
        
        # Handle cases where gradients might be None
        grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, self.model.trainable_variables)]
        
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Store loss for logging
        self.current_loss = loss

        # Calculate TD errors for priority updates
        errors = tf.abs(target_q - q_pred_actions)
        self.memory.update_priorities(indices, errors.numpy())

    def reduce_epsilon(self):
        # Decay epsilon after each episode
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path):
        # Ensure the path ends with '.keras' for the native Keras format
        if not path.endswith('.keras') and not path.endswith('.h5'):
            path += '.keras'
        self.model.save(path)  # Keras determines format based on extension

    def load_model(self, path):
        """
        Loads the model from the specified path, ensuring that custom objects are recognized.
        """
        self.model = tf.keras.models.load_model(
            path, 
            compile=False, 
            custom_objects={'combine_streams': combine_streams}
        )
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
