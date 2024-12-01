# backend/train_dqn.py

import os
import pickle
import numpy as np
import tensorflow as tf
from datetime import datetime
from game_2048_env import Game2048Env
from dqn_agent import DQNAgent, combine_streams  # Import combine_streams
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import mixed_precision

# 1. TensorFlow GPU Configuration
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Set mixed precision policy
mixed_precision.set_global_policy('mixed_float16')

# 2. Delete Existing Models and Agent States to Start Fresh
saved_model_path = "dqn_model_latest.keras"
best_model_path = "dqn_model_best.keras"
agent_state_path = "agent_state.pkl"

if os.path.exists(saved_model_path):
    os.remove(saved_model_path)
    print(f"Deleted existing model: {saved_model_path}")
if os.path.exists(best_model_path):
    os.remove(best_model_path)
    print(f"Deleted existing best model: {best_model_path}")
if os.path.exists(agent_state_path):
    os.remove(agent_state_path)
    print(f"Deleted existing agent state: {agent_state_path}")

# 3. Initialize Environment and Agent
print("Initializing environment and agent...")
env = Game2048Env()
state_shape = env.observation_space.shape  # e.g., (4, 4, 1)
action_size = env.action_space.n
print(f"State Shape: {state_shape}, Action Size: {action_size}")

# Initialize a new agent
agent = DQNAgent(state_shape, action_size)

# 4. Training Parameters
episodes = 5000  # Increased from 1000
max_steps = 3000  # To prevent infinite loops
log_frequency = 10  # Print summary every 10 episodes
max_checkpoints = 5  # Maximum number of checkpoints to keep
checkpoint_dir = "models/checkpoints/"
best_score = -np.inf
os.makedirs(checkpoint_dir, exist_ok=True)

# 5. Setup TensorBoard
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    profile_batch='50,60'  # Optional: profile specific batches
)

print("Starting training...")

# Initialize moving averages
moving_average_score = 0
moving_average_max_tile = 0

# 6. Training Loop
for e in range(1, episodes + 1):
    print(f"\nStarting Episode {e}")
    state = env.reset()
    done = False
    total_reward = 0
    max_tile = 0
    step = 0
    no_progress_steps = 0  # Initialize no progress counter
    action_counts = [0] * action_size  # Initialize action counts

    while not done and step < max_steps:
        prev_board = env.board.copy()
        action = agent.act(state)
        action_counts[action] += 1
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        max_tile = max(max_tile, np.max(env.board))

        # Check for progress
        if np.array_equal(prev_board, env.board):
            no_progress_steps += 1
            if no_progress_steps >= 30:
                print("No progress made for 30 steps. Terminating episode.")
                done = True
                # Apply penalty for termination
                total_reward -= 5
        else:
            no_progress_steps = 0  # Reset counter if progress is made

        agent.remember(state, action, reward, next_state, done)
        state = next_state

        agent.replay()

        step += 1

        # Update target network every 1000 steps
        if step % 1000 == 0:
            print(f"Step {step}: Updating target model.")
            agent.update_target_model()

    # Reduce exploration rate
    agent.reduce_epsilon()
    print(f"Episode {e}: Epsilon reduced to {agent.epsilon}")

    # Update target network after each episode as a backup
    agent.update_target_model()
    print(f"Score: {total_reward} | Max Tile: {int(max_tile)}")

    # Update moving averages
    moving_average_score = (moving_average_score * 0.99) + (total_reward * 0.01)
    moving_average_max_tile = (moving_average_max_tile * 0.99) + (max_tile * 0.01)

    # Log metrics to TensorBoard
    with tf.summary.create_file_writer(log_dir).as_default():
        tf.summary.scalar('Score', total_reward, step=e)
        tf.summary.scalar('Max Tile', max_tile, step=e)
        tf.summary.scalar('Epsilon', agent.epsilon, step=e)
        tf.summary.scalar('Loss', agent.current_loss, step=e)
        tf.summary.scalar('Moving_Average_Score', moving_average_score, step=e)
        tf.summary.scalar('Moving_Average_Max_Tile', moving_average_max_tile, step=e)
        for i in range(action_size):
            tf.summary.scalar(f'Action_{i}_Count', action_counts[i], step=e)

    # Update and save best model
    if total_reward > best_score:
        best_score = total_reward
        agent.save_model(best_model_path)
        print(f"New best score: {best_score} | Max Tile: {int(max_tile)}. Model saved.")

    # Save the latest model and agent state at specified intervals
    if e % log_frequency == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"dqn_checkpoint_{e}.keras")
        agent.save_model(checkpoint_path)
        with open(agent_state_path, 'wb') as f:
            pickle.dump({'epsilon': agent.epsilon, 'beta': agent.beta}, f)
        print(f"Episode {e}/{episodes} | Score: {int(total_reward)} | Max Tile: {int(max_tile)} | Epsilon: {agent.epsilon:.4f}")

        # Manage the number of checkpoints
        checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("dqn_checkpoint_")])
        if len(checkpoint_files) > max_checkpoints:
            oldest_checkpoint = checkpoint_files[0]
            os.remove(os.path.join(checkpoint_dir, oldest_checkpoint))
            print(f"Removed oldest checkpoint: {oldest_checkpoint}")

    # Early Stopping if target is reached
    if max_tile >= 2048:
        print(f"Reached target tile at Episode {e}. Stopping training.")
        break

# 7. Final Save after all episodes
agent.save_model(saved_model_path)
with open(agent_state_path, 'wb') as f:
    pickle.dump({'epsilon': agent.epsilon, 'beta': agent.beta}, f)
print("Training completed and final model saved.")
