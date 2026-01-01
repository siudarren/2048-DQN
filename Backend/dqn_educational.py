# dqn_educational.py
"""
Deep Q-Network (DQN) for 2048 - Educational Version
====================================================

This implementation is designed to teach you DQN step-by-step.
Each section is heavily commented to explain what's happening and why.

Key Concepts:
1. Q-Learning: Learn value of state-action pairs
2. Deep Neural Network: Approximate Q-function
3. Experience Replay: Learn from past experiences
4. Target Network: Stabilize training
5. Epsilon-Greedy: Balance exploration vs exploitation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


# ==============================================================================
# PART 1: THE Q-NETWORK (Brain of the Agent)
# ==============================================================================

class QNetwork(nn.Module):
    """
    The Q-Network is a neural network that learns to predict Q-values.
    
    Input: Game state (4x4 grid)
    Output: Q-values for each action [up, down, left, right]
    
    Architecture:
    - Convolutional layers: Extract spatial patterns (like "tiles in corner")
    - Fully connected layers: Combine patterns to decide actions
    """
    
    def __init__(self):
        super(QNetwork, self).__init__()
        
        # === Convolutional Layers ===
        # Why CNN? 2048 has spatial structure - nearby tiles matter!
        # Conv layer 1: Look for small patterns (2x2 regions)
        self.conv1 = nn.Conv2d(
            in_channels=16,      # Input: 1 channel (grayscale-like board)
            out_channels=128,   # Output: 128 feature maps
            kernel_size=2,      # Look at 2x2 regions
            stride=1,
            padding=0
        )
        # After conv1: 4x4 → 3x3
        
        # Conv layer 2: Look for larger patterns
        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=2,
            stride=1,
            padding=0
        )
        # After conv2: 3x3 → 2x2
        
        # === Fully Connected Layers ===
        # Flatten the 2x2x128 = 512 features
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Output layer: 4 Q-values (one per action)
        self.fc3 = nn.Linear(256, 4)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Initialize weights (important for stable training)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass: state → Q-values
        
        Args:
            x: State tensor [batch_size, 1, 4, 4]
        
        Returns:
            Q-values [batch_size, 4]
        """
        # Print shapes for learning (comment out after understanding)
        # print(f"Input shape: {x.shape}")
        
        # Convolutional layers with ReLU activation
        x = self.relu(self.conv1(x))  # [batch, 128, 3, 3]
        # print(f"After conv1: {x.shape}")
        
        x = self.relu(self.conv2(x))  # [batch, 128, 2, 2]
        # print(f"After conv2: {x.shape}")
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)     # [batch, 512]
        # print(f"After flatten: {x.shape}")
        
        # Fully connected layers
        x = self.relu(self.fc1(x))    # [batch, 256]
        x = self.relu(self.fc2(x))    # [batch, 256]
        
        # Output Q-values (no activation - can be negative!)
        q_values = self.fc3(x)        # [batch, 4]
        # print(f"Q-values shape: {q_values.shape}")
        
        return q_values

class ValueNetwork(nn.Module):
    """
    Afterstate Value Network:
    Input : (16, 4, 4) planes
    Output: scalar V(s')
    Shares the same conv trunk as QNetwork, but final layer is size 1.
    """
    def __init__(self):
        super(ValueNetwork, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=16,
            out_channels=128,
            kernel_size=2,
            stride=1,
            padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=2,
            stride=1,
            padding=0
        )

        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 256)
        # ONE output instead of 4 actions
        self.fc3 = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [batch, 16, 4, 4]
        x = self.relu(self.conv1(x))      # [batch, 128, 3, 3]
        x = self.relu(self.conv2(x))      # [batch, 128, 2, 2]
        x = x.view(x.size(0), -1)         # [batch, 512]
        x = self.relu(self.fc1(x))        # [batch, 256]
        x = self.relu(self.fc2(x))        # [batch, 256]
        v = self.fc3(x)                   # [batch, 1]
        return v.squeeze(-1)              # [batch]

# ==============================================================================
# PART 2: EXPERIENCE REPLAY (Memory of the Agent)
# ==============================================================================

class ReplayBuffer:
    """
    Experience Replay Buffer stores past experiences (transitions).
    
    Why do we need this?
    - Breaks correlation between consecutive samples
    - Allows learning from rare but important events multiple times
    - Makes learning more stable
    
    A transition is: (state, action, reward, next_state, done)
    """
    
    def __init__(self, capacity=100000):
        """
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """
        Store a transition.
        
        Args:
            state: Current state (4x4 board)
            action: Action taken (0-3)
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a random batch of transitions.
        
        Why random? To break temporal correlations!
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of numpy arrays: (states, actions, rewards, next_states, dones)
        """
        # Sample random transitions
        batch = random.sample(self.buffer, batch_size)
        
        # Unzip the batch
        state, action, reward, next_state, done = zip(*batch)
        
        # Convert to numpy arrays
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.float32)
        )
    
    def __len__(self):
        """Return current buffer size"""
        return len(self.buffer)

class AfterstateReplayBuffer:
    """
    Replay buffer for afterstate value learning.
    Transition: (after_state, reward, next_after_state, done)
    """
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, after_state, reward, next_after_state, done):
        self.buffer.append((after_state, reward, next_after_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        after_state, reward, next_after_state, done = zip(*batch)
        return (
            np.array(after_state),
            np.array(reward, dtype=np.float32),
            np.array(next_after_state),
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ==============================================================================
# PART 3: THE DQN AGENT (Putting it all together)
# ==============================================================================

class DQNAgent:
    """
    DQN Agent that learns to play 2048.
    
    Key Components:
    1. Policy Network: Current Q-function being trained
    2. Target Network: Stable Q-function for computing targets
    3. Replay Buffer: Stores past experiences
    4. Optimizer: Updates network weights
    """
    
    def __init__(self, 
                 learning_rate=1e-4,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 batch_size=64,
                 target_update_freq=1000):
        """
        Initialize DQN Agent.
        
        Hyperparameters explained:
        - learning_rate: How fast to update weights (too high = unstable, too low = slow)
        - gamma: Discount factor for future rewards (0-1, higher = care more about future)
        - epsilon_start: Initial exploration rate (1.0 = 100% random)
        - epsilon_end: Final exploration rate (0.01 = 1% random)
        - epsilon_decay: How fast to reduce exploration
        - batch_size: Number of experiences to learn from at once
        - target_update_freq: How often to update target network
        """
        
        # === Device Setup ===
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("🖥️  Using device: MPS (Apple GPU)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("🖥️  Using device: CUDA GPU")
        else:
            self.device = torch.device("cpu")
            print("🖥️  Using device: CPU")

        
        # === Neural Networks ===
        # Policy network: The one we're actively training
        self.policy_net = QNetwork().to(self.device)
        
        # Target network: Stable copy for computing TD targets
        # Why? Prevents "chasing a moving target" problem
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Always in evaluation mode
        
        print(f"\n📊 Network Architecture:")
        print(f"    Total parameters: {sum(p.numel() for p in self.policy_net.parameters()):,}")
        
        # === Optimizer ===
        # Adam optimizer - adaptive learning rate
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # === Hyperparameters ===
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # === Replay Buffer ===
        self.memory = ReplayBuffer(capacity=100000)
        
        # === Training Statistics ===
        self.steps = 0
        self.episodes = 0
        self.losses = []
    
    def select_action(self, state, valid_actions=None, training=True):
        """
        Select action with Masking.
        valid_actions: List of valid indices, e.g., [1, 2] for Down/Left
        """
        
        # If no mask provided (or all moves valid), fallback to standard
        if valid_actions is None or len(valid_actions) == 0:
            valid_actions = [0, 1, 2, 3]

        # === EXPLORATION ===
        if training and random.random() < self.epsilon:
            # Only pick from VALID random actions
            action = random.choice(valid_actions)
            return action
        
        # === EXPLOITATION ===
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor) # [1, 4]
            
            # --- MASKING MAGIC ---
            # Create a mask of -infinity
            full_mask = torch.full_like(q_values, float('-inf'))
            
            # Allow only valid actions
            # We map the list [0, 2] to tensor indices
            valid_tensor = torch.tensor(valid_actions, device=self.device)
            full_mask[0, valid_tensor] = 0 
            
            # Add mask to Q-values (Valid stay same, Invalid become -inf)
            masked_q_values = q_values + full_mask
            
            # Choose action
            action = masked_q_values.argmax().item()
            return action
    
    def train_step(self):
        """
        Perform one training step (update the Q-network).
        
        This is where the magic happens!
        
        Algorithm:
        1. Sample batch from replay buffer
        2. Compute current Q-values using policy network
        3. Compute target Q-values using target network
        4. Compute loss (Mean Squared Error)
        5. Update policy network weights
        
        Returns:
            loss: Training loss (float), or None if not enough samples
        """
        
        # === CHECK BUFFER SIZE ===
        if len(self.memory) < self.batch_size:
            return None  # Not enough samples yet
        
        # === STEP 1: Sample Batch ===
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        

        # Convert to PyTorch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # === STEP 2: Compute Current Q-Values ===
        # Q(s, a) from policy network
        current_q_values = self.policy_net(states)  # [batch_size, 4]
        
        # Select Q-values for actions that were actually taken
        # gather() picks Q-values at indices specified by actions
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        # === STEP 3: Compute Target Q-Values ===
        # Target = r + gamma * max_a' Q_target(s', a')
        # But if episode ended (done=1), target = r only
        
        with torch.no_grad():  # Don't compute gradients for target
            # Get Q-values for next states from TARGET network
            next_q_values = self.target_net(next_states)  # [batch_size, 4]
            
            # Take maximum Q-value for each next state
            max_next_q_values = next_q_values.max(1)[0]  # [batch_size]
            
            # Compute targets using Bellman equation
            # If done, target = reward only
            # If not done, target = reward + gamma * max future Q
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        # === STEP 4: Compute Loss ===
        # Mean Squared Error between current Q and target Q
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # === STEP 5: Update Policy Network ===
        self.optimizer.zero_grad()  # Clear old gradients
        loss.backward()             # Compute gradients
        
        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        
        self.optimizer.step()       # Update weights
        
        # === Update Target Network ===
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"🎯 Target network updated at step {self.steps}")
        
        # Store loss for logging
        self.losses.append(loss.item())
        
        return loss.item()
    
    def update_epsilon(self):
        """
        Decay epsilon (reduce exploration over time).
        
        As agent learns, it should explore less and exploit more.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
        }, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        print(f"📂 Model loaded from {filepath}")


def random_symmetry_pair(state1: np.ndarray, state2: np.ndarray) -> tuple:
    """
    Apply the SAME random rotation/flip to TWO states.
    Critical for maintaining (s, s') relationship in replay buffer.
    
    Args:
        state1: (C, 4, 4) first state
        state2: (C, 4, 4) second state
    
    Returns:
        (aug_state1, aug_state2) with same transformation applied
    """
    # Choose random transformation
    k = np.random.randint(0, 4)  # 0, 90, 180, or 270 degrees
    flip = np.random.rand() < 0.5  # 50% chance of horizontal flip
    
    # Apply SAME transformation to both
    aug1 = np.rot90(state1, k, axes=(-2, -1))
    aug2 = np.rot90(state2, k, axes=(-2, -1))
    
    if flip:
        aug1 = np.flip(aug1, axis=-1)
        aug2 = np.flip(aug2, axis=-1)
    
    return aug1.copy(), aug2.copy()

class AfterstateAgent:
    """
    Afterstate value-learning agent for 2048.

    Learns V(s') where s' = board after the move (before random tile).
    Bellman update:
        V(s_prev') <- r_t + gamma * V(s_t')  (TD(0) between afterstates)
    """
    def __init__(self,
                 learning_rate=1e-4,
                 gamma=1.0,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 batch_size=64,
                 target_update_freq=1000):

        # === Device ===
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("🖥️  Using device: MPS (Apple GPU)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("🖥️  Using device: CUDA GPU")
        else:
            self.device = torch.device("cpu")
            print("🖥️  Using device: CPU")

        # === Networks ===
        self.policy_net = ValueNetwork().to(self.device)
        self.target_net = ValueNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        print("\n📊 Afterstate Value Network:")
        print(f"    Total parameters: {sum(p.numel() for p in self.policy_net.parameters()):,}")

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.memory = AfterstateReplayBuffer(capacity=100000)

        self.steps = 0
        self.episodes = 0
        self.losses = []

    # Simple value evaluation helper (batch: [B,16,4,4])
    def value(self, after_state_batch):
        with torch.no_grad():
            x = torch.FloatTensor(after_state_batch).to(self.device)
            return self.policy_net(x).cpu().numpy()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        after_states, rewards, next_after_states, dones = self.memory.sample(self.batch_size)
        # Apply the same random symmetry to (s, s') for each transition
        for i in range(after_states.shape[0]):
            aug_s, aug_ns = random_symmetry_pair(after_states[i], next_after_states[i])
            after_states[i]      = aug_s
            next_after_states[i] = aug_ns

        after_states = torch.FloatTensor(after_states).to(self.device)          # [B,16,4,4]
        next_after_states = torch.FloatTensor(next_after_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)                   # [B]
        dones = torch.FloatTensor(dones).to(self.device)                       # [B]

        # Current values V(s_prev')
        current_v = self.policy_net(after_states)                              # [B]

        with torch.no_grad():
            next_v = self.target_net(next_after_states)                        # [B]
            target_v = rewards + (1.0 - dones) * self.gamma * next_v

        loss = F.mse_loss(current_v, target_v)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"🎯 Afterstate target net updated at step {self.steps}")

        self.losses.append(loss.item())
        return loss.item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
        }, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        print(f"📂 Model loaded from {filepath}")

# ==============================================================================
# PART 4: HELPER FUNCTIONS
# ==============================================================================

def print_training_summary(agent):
    """Print summary of agent's current state"""
    print("\n" + "="*70)
    print("📈 TRAINING SUMMARY")
    print("="*70)
    print(f"Episodes completed: {agent.episodes}")
    print(f"Training steps: {agent.steps}")
    print(f"Current epsilon: {agent.epsilon:.4f}")
    print(f"Replay buffer size: {len(agent.memory):,}")
    if agent.losses:
        print(f"Average loss (last 100): {np.mean(agent.losses[-100:]):.4f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    """
    Quick test to verify everything works
    """
    print("🧪 Testing DQN Components...\n")
    
    # Test Q-Network
    print("1️⃣ Testing Q-Network...")
    net = QNetwork()
    dummy_state = torch.randn(1, 16, 4, 4)  # Random 4x4 board
    q_values = net(dummy_state)
    print(f"   Input shape: {dummy_state.shape}")
    print(f"   Output Q-values: {q_values}")
    print(f"   ✅ Q-Network works!\n")
    
    # Test Replay Buffer
    print("2️⃣ Testing Replay Buffer...")
    buffer = ReplayBuffer(capacity=1000)
    for i in range(100):
        state = np.random.rand(16, 4, 4).astype(np.float32)
        action = random.randint(0, 3)
        reward = random.random()
        next_state = np.random.rand(16, 4, 4).astype(np.float32)
        done = random.choice([True, False])
        buffer.push(state, action, reward, next_state, done)
    
    batch = buffer.sample(32)
    print(f"   Buffer size: {len(buffer)}")
    print(f"   Batch shapes: states={batch[0].shape}, actions={batch[1].shape}")
    print(f"   ✅ Replay Buffer works!\n")
    
    # Test DQN Agent
    print("3️⃣ Testing DQN Agent...")
    agent = DQNAgent()
    
    # Test action selection
    state = np.random.rand(1, 4, 4)
    action = agent.select_action(state)
    print(f"   Selected action: {action}")
    
    # Add some experiences
    for i in range(100):
        state = np.random.rand(1, 4, 4)
        action = random.randint(0, 3)
        reward = random.random()
        next_state = np.random.rand(1, 4, 4)
        done = False
        agent.memory.push(state, action, reward, next_state, done)
    
    # Test training step
    loss = agent.train_step()
    print(f"   Training loss: {loss:.4f}")
    print(f"   ✅ DQN Agent works!\n")
    
    print("🎉 All components working! Ready to train on 2048!")

    # Add this method to your DQNAgent class in dqn_educational.py

def select_actions_batch(self, state_batch, training=True):
    """
    Select actions for a batch of states (for parallel environments).
    
    Args:
        state_batch: Numpy array of shape (batch_size, 1, 4, 4)
        training: Whether to use epsilon-greedy
    
    Returns:
        actions: List of integers [action1, action2, ...]
    """
    batch_size = state_batch.shape[0]
    actions = []
    
    # === EXPLORATION ===
    if training:
        # For each state, decide explore or exploit
        explore_mask = np.random.random(batch_size) < self.epsilon
        
        # === EXPLOITATION (for non-exploring states) ===
        if not explore_mask.all():
            with torch.no_grad():
                # Convert to tensor and send to device
                state_tensor = torch.FloatTensor(state_batch).to(self.device)
                q_values = self.policy_net(state_tensor)  # (batch_size, 4)
                best_actions = q_values.argmax(dim=1).cpu().numpy()
        
        # Combine exploration and exploitation
        for i in range(batch_size):
            if explore_mask[i]:
                # Explore: random action
                actions.append(np.random.randint(0, 4))
            else:
                # Exploit: best action
                actions.append(int(best_actions[i]))
    
    # === PURE EXPLOITATION (no exploration) ===
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_batch).to(self.device)
            q_values = self.policy_net(state_tensor)
            best_actions = q_values.argmax(dim=1).cpu().numpy()
            actions = best_actions.tolist()
    
    return actions


# Alternative with action masking (better version):

def select_actions_batch_masked(self, state_batch, valid_actions_list, training=True):
    """
    Select actions for a batch of states with action masking.
    
    Args:
        state_batch: Numpy array of shape (batch_size, 1, 4, 4)
        valid_actions_list: List of lists, e.g., [[0,1,2], [1,3], [0,1,2,3], ...]
        training: Whether to use epsilon-greedy
    
    Returns:
        actions: List of integers
    """
    batch_size = state_batch.shape[0]
    actions = []
    
    # === EXPLORATION ===
    if training:
        explore_mask = np.random.random(batch_size) < self.epsilon
        
        # Get Q-values for all states
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_batch).to(self.device)
            q_values = self.policy_net(state_tensor)  # (batch_size, 4)
        
        # For each state, pick action
        for i in range(batch_size):
            valid_actions = valid_actions_list[i]
            
            if not valid_actions:
                # No valid actions - shouldn't happen
                actions.append(0)
                continue
            
            if explore_mask[i]:
                # Explore: random valid action
                actions.append(np.random.choice(valid_actions))
            else:
                # Exploit: best valid action
                # Mask invalid actions
                masked_q = q_values[i].clone()
                mask = torch.ones(4, device=self.device) * float('-inf')
                mask[valid_actions] = 0
                masked_q = masked_q + mask
                actions.append(int(masked_q.argmax()))
    
    else:
        # Pure exploitation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_batch).to(self.device)
            q_values = self.policy_net(state_tensor)
        
        for i in range(batch_size):
            valid_actions = valid_actions_list[i]
            
            if not valid_actions:
                actions.append(0)
                continue
            
            # Best valid action
            masked_q = q_values[i].clone()
            mask = torch.ones(4, device=self.device) * float('-inf')
            mask[valid_actions] = 0
            masked_q = masked_q + mask
            actions.append(int(masked_q.argmax()))
    
    return actions

    import numpy as np

def board_to_planes(board, max_exp=15):
    """
    board: 4x4 array of tile values (0, 2, 4, 8, ..., 32768)
    returns: (16, 4, 4) binary planes
    channel 0: empty cells
    channel 1: cells with 2
    channel 2: cells with 4
    ...
    channel 15: cells with 32768
    """
    planes = np.zeros((max_exp + 1, 4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            v = board[i, j]
            if v == 0:
                planes[0, i, j] = 1.0
            else:
                e = int(np.log2(v))  # 2→1, 4→2, 8→3, ...
                e = min(e, max_exp)
                planes[e, i, j] = 1.0
    return planes