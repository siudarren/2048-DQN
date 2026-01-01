# train_dqn_parallel.py
"""
Parallel DQN Training for 2048
================================

Uses multiple environments running in parallel to collect experiences faster.

Key idea: Instead of 1 env collecting 1 experience per step,
          we have N envs collecting N experiences per step!
"""

import sys

import os
import time
import random
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import torch

from game_2048_env import Game2048Env
from dqn_educational import DQNAgent, board_to_planes, AfterstateAgent

def select_action_with_mask(agent, state, valid_actions, training=True):
    """
    Epsilon-greedy over *valid* actions only.

    Args:
        agent: DQNAgent
        state: np.array shape (1, 4, 4)
        valid_actions: list of ints, subset of {0, 1, 2, 3}
        training: bool

    Returns:
        int action in valid_actions
    """
    # Exploration
    if training and random.random() < agent.epsilon:
        return random.choice(valid_actions)

    # Exploitation: use Q-network, but restrict to valid_actions
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)  # [1, 16, 4, 4]
        q_values = agent.policy_net(state_t)[0].cpu().numpy()             # [4]

    best_action = max(valid_actions, key=lambda a: q_values[a])
    return int(best_action)

def select_action_afterstate(agent, env, state_planes, valid_actions, training=True):
    """
    Choose action by evaluating r + gamma * V(afterstate) over valid actions.
    
    Args:
        state_planes: (16, 4, 4) current observation (unused here, but good for API consistency)
        valid_actions: List of valid action indices
    """
    gamma = agent.gamma

    # === EXPLORATION (Epsilon-Greedy) ===
    if training and random.random() < agent.epsilon:
        return random.choice(valid_actions)

    # === EXPLOITATION (Greedy via Afterstates) ===
    best_action = None
    best_value = -float("inf")

    # Evaluate every possible move
    for action in valid_actions:
        # Get the board state immediately AFTER the move, but BEFORE the random tile
        # Note: Your environment needs to support `get_afterstate(action)`
        moved, merge_reward, after_planes = env.get_afterstate(action)
        
        if not moved:
            continue

        # Evaluate V(s') using the neural network
        # We need to wrap input in a batch dimension: [1, 16, 4, 4]
        after_tensor = torch.FloatTensor(after_planes).unsqueeze(0).to(agent.device)
        
        with torch.no_grad():
            v_next = agent.policy_net(after_tensor).item()

        # Q(s, a) = Reward + Gamma * V(afterstate)
        q_value = merge_reward + gamma * v_next

        if q_value > best_value:
            best_value = q_value
            best_action = action

    # Fallback if something goes wrong (shouldn't happen with valid_actions)
    if best_action is None:
        return random.choice(valid_actions)

    return int(best_action)
    
def train_dqn_parallel(num_episodes=5000, num_envs=16, updates_per_step=4):
    """
    Train DQN with parallel environments.

    Args:
        num_episodes: total episodes to train (across ALL envs)
        num_envs: number of parallel environments
        updates_per_step: how many training updates per env step
    """

    print("\n" + "="*70)
    print("🚀 PARALLEL DQN TRAINING FOR 2048")
    print("="*70)
    print(f"Number of parallel environments: {num_envs}")
    print(f"Training updates per step: {updates_per_step}")
    print(f"Target episodes: {num_episodes}")
    print()

    # === Setup ===
    print("📋 Setting up environments and agent...")

    envs = [Game2048Env() for _ in range(num_envs)]

    agent = DQNAgent(
        learning_rate=1e-4,
        gamma=1,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        target_update_freq=1000
    )

    agent.load("models/dqn_2048_parallel_ep13000.pt")

    print(f"✅ Created {num_envs} environments")
    print(f"✅ Agent ready\n")

    # === Tracking metrics ===
    episode_rewards = []
    episode_max_tiles = []
    episode_scores = []
    episode_lengths = []

    episode_valid_moves = []
    episode_invalid_moves = []  # will stay ~0 because we mask invalid actions

    recent_rewards = deque(maxlen=100)
    recent_max_tiles = deque(maxlen=100)

    # Per-environment state & counters
    states = [env.reset() for env in envs] 
    episode_step_counts = [0] * num_envs
    episode_reward_sums = [0.0] * num_envs
    episode_valid_counts = [0] * num_envs
    episode_invalid_counts = [0] * num_envs  # kept for completeness

    total_episodes_completed = 0
    start_time = time.time()
    step = 0

    print("🚀 Starting parallel training...\n" + "="*70 + "\n")

    # === Main training loop ===
    while total_episodes_completed < num_episodes:
        step += 1

        # Step each environment once
        for i, env in enumerate(envs):
            if total_episodes_completed >= num_episodes:
                break

            state = states[i]

            # 1. Get valid actions for this env
            valid_actions = env.get_valid_actions()

            # If no valid actions: treat as terminal, log episode, reset
            if not valid_actions:
                max_tile = env.board.max()
                score = env.score

                episode_rewards.append(episode_reward_sums[i])
                episode_max_tiles.append(max_tile)
                episode_scores.append(score)
                episode_lengths.append(episode_step_counts[i])
                episode_valid_moves.append(episode_valid_counts[i])
                episode_invalid_moves.append(episode_invalid_counts[i])

                recent_rewards.append(episode_reward_sums[i])
                recent_max_tiles.append(max_tile)
                total_episodes_completed += 1

                # Reset env & counters
                states[i] = env.reset()
                episode_step_counts[i] = 0
                episode_reward_sums[i] = 0.0
                episode_valid_counts[i] = 0
                episode_invalid_counts[i] = 0

                # Epsilon decay once per completed episode
                agent.update_epsilon()
                agent.episodes = total_episodes_completed

                continue

            # 2. Choose action (masked epsilon-greedy)
            action = select_action_with_mask(agent, state, valid_actions, training=True)

            # All chosen actions are valid by construction
            episode_valid_counts[i] += 1

            # 3. Step environment
            next_state, reward, done, info = env.step(action)

            agent.memory.push(state, action, reward, next_state, done)

            # 5. Update counters
            episode_step_counts[i] += 1
            episode_reward_sums[i] += reward

            # 6. Handle episode end
            if done:
                max_tile = env.board.max()
                score = env.score

                episode_rewards.append(episode_reward_sums[i])
                episode_max_tiles.append(max_tile)
                episode_scores.append(score)
                episode_lengths.append(episode_step_counts[i])
                episode_valid_moves.append(episode_valid_counts[i])
                episode_invalid_moves.append(episode_invalid_counts[i])

                recent_rewards.append(episode_reward_sums[i])
                recent_max_tiles.append(max_tile)
                total_episodes_completed += 1

                # Reset env & counters
                states[i] = env.reset()
                episode_step_counts[i] = 0
                episode_reward_sums[i] = 0.0
                episode_valid_counts[i] = 0
                episode_invalid_counts[i] = 0

                # Epsilon decay once per completed episode
                agent.episodes = total_episodes_completed
                if total_episodes_completed % 5000 == 1:
                    agent.epsilon = 0.2
                else:
                    agent.update_epsilon()

                states[i] = env.reset()
            else:
                # Continue episode
                states[i] = next_state

        # === Training updates ===
        if len(agent.memory) >= agent.batch_size:
            for _ in range(updates_per_step):
                agent.train_step()

        # === Logging ===
        if total_episodes_completed > 0 and total_episodes_completed % 10 == 0:
            elapsed = time.time() - start_time
            episodes_per_sec = total_episodes_completed / elapsed

            avg_valid = np.mean(episode_valid_moves[-100:]) if episode_valid_moves else 0
            avg_invalid = np.mean(episode_invalid_moves[-100:]) if episode_invalid_moves else 0
            total_mv = avg_valid + avg_invalid
            valid_pct = (avg_valid / total_mv * 100) if total_mv > 0 else 0

            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            avg_max_tile = np.mean(recent_max_tiles) if recent_max_tiles else 0
            avg_score = np.mean(episode_scores[-100:]) if episode_scores else 0
            avg_length = np.mean(episode_lengths[-100:]) if episode_lengths else 0
            avg_loss = np.mean(agent.losses[-100:]) if agent.losses else 0

            print(f"Episode {total_episodes_completed}/{num_episodes} "
                  f"({episodes_per_sec:.1f} ep/s)")
            print("  📊 Performance:")
            print(f"     Reward: {episode_rewards[-1]:.1f} (avg: {avg_reward:.1f})")
            print(f"     Max tile: {episode_max_tiles[-1]} (avg: {avg_max_tile:.1f})")
            print(f"     Score: {episode_scores[-1]:.0f} (avg: {avg_score:.1f})")
            print(f"     Moves: {episode_lengths[-1]} (avg: {avg_length:.1f})")
            print(f"     Valid: {avg_valid:.1f} ({valid_pct:.1f}%)")
            print(f"     Invalid: {avg_invalid:.1f} ({100 - valid_pct:.1f}%)")
            print("  🎯 Training:")
            print(f"     Epsilon: {agent.epsilon:.4f}")
            print(f"     Loss: {avg_loss:.4f}")
            print(f"     Buffer: {len(agent.memory):,}/{agent.memory.capacity:,}")
            print(f"     Steps: {agent.steps:,}")
            print(f"  ⏱️  Time: {elapsed/60:.1f} min")
            print("-" * 70)

        # === Milestones ===
        if total_episodes_completed in [100, 500, 1000, 2000] and total_episodes_completed > 0:
            print(f"\n🎉 MILESTONE: {total_episodes_completed} episodes completed!")
            last = min(100, len(episode_max_tiles))
            print(f"   Average max tile (last {last}): {np.mean(episode_max_tiles[-last:]):.1f}")
            print("-" * 70 + "\n")

        # === Save Model ===
        if total_episodes_completed > 0 and total_episodes_completed % 500 == 0:
            os.makedirs('models', exist_ok=True)
            agent.save(f'models/dqn_2048_parallel_ep{total_episodes_completed}.pt')
            plot_progress(
                episode_rewards,
                episode_max_tiles,
                episode_scores,
                episode_lengths,
                total_episodes_completed
            )

    # === Training complete ===
    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Total episodes: {num_episodes}")
    print(f"Total steps: {agent.steps:,}")
    print(f"Episodes per second: {num_episodes / total_time:.2f}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print("\n📈 Final Performance (last 100 episodes):")
    last = min(100, len(episode_max_tiles))
    print(f"   Average max tile: {np.mean(episode_max_tiles[-last:]):.1f}")
    print(f"   Best max tile: {max(episode_max_tiles[-last:])}")
    print(f"   Average score: {np.mean(episode_scores[-last:]):.1f}")
    print(f"   Best score: {max(episode_scores[-last:])}")
    print("="*70 + "\n")

    agent.save('models/dqn_2048_parallel_final.pt')
    plot_progress(
        episode_rewards,
        episode_max_tiles,
        episode_scores,
        episode_lengths,
        num_episodes
    )

    return agent, episode_rewards, episode_max_tiles, episode_scores

def train_afterstate_parallel(num_episodes=5000, num_envs=16, updates_per_step=4):
    print("\n" + "="*70)
    print("🚀 PARALLEL AFTERSTATE VALUE TRAINING FOR 2048")
    print("="*70)

    start_time = time.time()

    # === Create envs and agent ===
    envs = [Game2048Env() for _ in range(num_envs)]
    agent = AfterstateAgent(
        learning_rate=5e-5,
        gamma=1,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9999,
        batch_size=128,
        target_update_freq=500,
    )
    
    # Optional: Load checkpoint
    if os.path.exists("models/afterstate_2048_parallel_ep44000.pt"):
        agent.load("models/afterstate_2048_parallel_ep44000.pt")
        print(f"✅ Loaded checkpoint from episode {agent.episodes}")
    agent.epsilon = 0.1
    
    # === Per-env runtime state ===
    states = [env.reset() for env in envs]
    episode_reward_sums = [0.0] * num_envs
    episode_step_counts = [0] * num_envs
    last_afterstate = [None] * num_envs

    start_total_episodes = agent.episodes
    total_episodes_completed = agent.episodes  # Resume from checkpoint

    # === Global tracking for logging / plotting ===
    episode_rewards = []
    episode_scores = []
    episode_max_tiles = []
    episode_lengths = []

    recent_rewards = deque(maxlen=100)
    recent_max_tiles = deque(maxlen=100)

    print("🚀 Starting parallel AFTERSTATE training...\n" + "="*70 + "\n")

    while total_episodes_completed < num_episodes:
        # Step each environment
        for i, env in enumerate(envs):
            if total_episodes_completed >= num_episodes:
                break

            state_planes = states[i]
            valid_actions = env.get_valid_actions()

            # === If no valid actions: game over for this env ===
            if not valid_actions:
                max_tile = env.board.max()
                score = env.score

                # Log this episode
                episode_rewards.append(episode_reward_sums[i])
                episode_scores.append(score)
                episode_max_tiles.append(max_tile)
                episode_lengths.append(episode_step_counts[i])

                recent_rewards.append(episode_reward_sums[i])
                recent_max_tiles.append(max_tile)

                total_episodes_completed += 1
                agent.episodes = total_episodes_completed
                agent.update_epsilon()

                # Reset env state and counters
                states[i] = env.reset()
                episode_reward_sums[i] = 0.0
                episode_step_counts[i] = 0
                last_afterstate[i] = None
                continue

            # === Choose action and get afterstate (ONCE!) ===
            action, curr_after, merge_reward = select_action_afterstate_efficient(
                agent, env, state_planes, valid_actions, training=True
            )

            # === Push transition from PREVIOUS step ===
            if last_afterstate[i] is not None:
                # Transition: last_afterstate → curr_after with merge_reward
                agent.memory.push(
                    last_afterstate[i],
                    merge_reward,
                    curr_after,
                    False  # Not terminal (yet)
                )

            # === Apply env.step to add random tile ===
            next_state_planes, env_reward, done, info = env.step(action)

            # Update per-episode tracking
            episode_reward_sums[i] += env_reward
            episode_step_counts[i] += 1

            # === Handle episode termination ===
            if done:
                # Terminal transition: curr_after led to game over
                agent.memory.push(
                    curr_after,
                    0,
                    curr_after,
                    True
                )
                
                # Log episode
                max_tile = env.board.max()
                score = env.score

                episode_rewards.append(episode_reward_sums[i])
                episode_scores.append(score)
                episode_max_tiles.append(max_tile)
                episode_lengths.append(episode_step_counts[i])

                recent_rewards.append(episode_reward_sums[i])
                recent_max_tiles.append(max_tile)

                total_episodes_completed += 1
                agent.episodes = total_episodes_completed
                agent.update_epsilon()

                # Reset
                states[i] = env.reset()
                episode_reward_sums[i] = 0.0
                episode_step_counts[i] = 0
                last_afterstate[i] = None
            else:
                # Store for next iteration
                last_afterstate[i] = curr_after
                states[i] = next_state_planes

        # === Training updates ===
        if len(agent.memory) >= agent.batch_size:
            for _ in range(updates_per_step):
                agent.train_step()

        # === Logging every 10 episodes ===
        if total_episodes_completed > 0 and total_episodes_completed % 10 == 0:
            elapsed = time.time() - start_time
            episodes_per_sec = (total_episodes_completed - start_total_episodes) / elapsed if elapsed > 0 else 0

            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            avg_max_tile = np.mean(recent_max_tiles) if recent_max_tiles else 0
            avg_score = np.mean(episode_scores[-100:]) if episode_scores else 0
            avg_length = np.mean(episode_lengths[-100:]) if episode_lengths else 0
            avg_loss = np.mean(agent.losses[-100:]) if agent.losses else 0

            print(f"\rEp {total_episodes_completed}/{num_episodes} | "
                  f"Score: {avg_score:.0f} | "
                  f"Tile: {avg_max_tile:.1f} | "
                  f"Eps: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"{episodes_per_sec:.1f} ep/s", end="")

        # === Milestones ===
        if total_episodes_completed in [100, 500, 1000, 2000, 5000] and total_episodes_completed > 0:
            print(f"\n🎉 MILESTONE: {total_episodes_completed} episodes!")
            last = min(100, len(episode_max_tiles))
            if last > 0:
                print(f"   Avg max tile: {np.mean(episode_max_tiles[-last:]):.1f}")

        # === Save Model ===
        if total_episodes_completed > start_total_episodes and total_episodes_completed % 500 == 0:
            os.makedirs('models', exist_ok=True)
            agent.save(f'models/afterstate_2048_parallel_ep{total_episodes_completed}.pt')
            agent.save('models/afterstate_2048_parallel_latest.pt')
            plot_progress(
                episode_rewards,
                episode_max_tiles,
                episode_scores,
                episode_lengths,
                total_episodes_completed
            )

    # === Training complete ===
    total_time = time.time() - start_time

    print("\n\n" + "="*70)
    print("🎉 AFTERSTATE TRAINING COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Episodes per second: {(total_episodes_completed - agent.episodes) / total_time:.2f}")

    last = min(100, len(episode_max_tiles))
    if last > 0:
        print(f"\nFinal avg max tile: {np.mean(episode_max_tiles[-last:]):.1f}")
        print(f"Best max tile: {max(episode_max_tiles[-last:])}")

    agent.save('models/afterstate_2048_parallel_final.pt')
    plot_progress(episode_rewards, episode_max_tiles, episode_scores, episode_lengths, num_episodes)

    return agent, episode_rewards, episode_max_tiles, episode_scores


def plot_progress(rewards, max_tiles, scores, lengths, episode):
    """Generate and save training graphs."""
    os.makedirs('plots', exist_ok=True)
    
    # Use aggregation to make plots readable if we have too many points
    window = 100
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Progress - Episode {episode}', fontsize=16)

    # Helper to plot data + moving average
    def plot_metric(ax, data, title, ylabel):
        ax.plot(data, alpha=0.3, color='blue', label='Raw')
        if len(data) >= window:
            ma = np.convolve(data, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(data)), ma, color='red', linewidth=2, label=f'Avg ({window})')
        ax.set_title(title)
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plot_metric(axes[0, 0], rewards, 'Total Rewards', 'Reward')
    plot_metric(axes[0, 1], max_tiles, 'Max Tile Achieved', 'Tile Value')
    plot_metric(axes[1, 0], scores, 'Game Score', 'Score')
    plot_metric(axes[1, 1], lengths, 'Episode Length', 'Steps')

    plt.tight_layout()
    plt.savefig(f'plots/training_ep{episode}.png')
    plt.close()
def select_action_afterstate_efficient(agent, env, state_planes, valid_actions, training=True):
    """
    Returns: (action, afterstate_planes, merge_reward)
    Combines selection and computation to avoid duplicate get_afterstate calls.
    """
    gamma = agent.gamma

    if training and random.random() < agent.epsilon:
        action = random.choice(valid_actions)
        moved, merge_reward, after_planes = env.get_afterstate(action)
        return action, after_planes, merge_reward

    # Exploitation - evaluate all actions
    best_action = None
    best_after = None
    best_reward = None
    best_value = -float("inf")

    for action in valid_actions:
        moved, merge_reward, after_planes = env.get_afterstate(action)
        if not moved:
            continue

        after_tensor = torch.FloatTensor(after_planes).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            v_next = agent.policy_net(after_tensor).item()

        q_value = merge_reward + gamma * v_next

        if q_value > best_value:
            best_value = q_value
            best_action = action
            best_after = after_planes
            best_reward = merge_reward

    if best_action is None:
        action = random.choice(valid_actions)
        moved, merge_reward, after_planes = env.get_afterstate(action)
        return action, after_planes, merge_reward

    return best_action, best_after, best_reward

# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================
if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    print("\n" + "="*70)
    print("🚀 PARALLEL DQN TRAINING FOR 2048")
    print("="*70)
    print("\nBenefits of parallel training:")
    print("  • Faster episode collection")
    print("  • Better data diversity in replay buffer")
    print("  • More efficient GPU/CPU utilization")
    print("="*70)

    agent, rewards, max_tiles, scores = train_afterstate_parallel(
        num_episodes=100000,
        num_envs=16,
        updates_per_step=4,
    )
