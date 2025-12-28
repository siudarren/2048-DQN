# train_dqn_parallel.py
"""
Parallel DQN Training for 2048
================================

Uses multiple environments running in parallel to collect experiences faster.

Key idea: Instead of 1 env collecting 1 experience per step,
          we have N envs collecting N experiences per step!
"""

import sys
sys.path.append('Backend')

import os
import time
import random
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import torch

from game_2048_env import Game2048Env
from dqn_educational import DQNAgent, board_to_planes


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


def plot_progress(rewards, max_tiles, scores, lengths, episode):
    """Create training progress plots."""
    os.makedirs('plots', exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Rewards
    axes[0, 0].plot(rewards, alpha=0.3)
    if len(rewards) >= 100:
        ma = np.convolve(rewards, np.ones(100)/100, mode='valid')
        axes[0, 0].plot(range(99, len(rewards)), ma, linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].grid(True, alpha=0.3)

    # Max tiles
    axes[0, 1].plot(max_tiles, alpha=0.3)
    if len(max_tiles) >= 100:
        ma = np.convolve(max_tiles, np.ones(100)/100, mode='valid')
        axes[0, 1].plot(range(99, len(max_tiles)), ma, linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Max Tile')
    axes[0, 1].set_title('Max Tile Achieved')
    axes[0, 1].grid(True, alpha=0.3)

    # Scores
    axes[1, 0].plot(scores, alpha=0.3)
    if len(scores) >= 100:
        ma = np.convolve(scores, np.ones(100)/100, mode='valid')
        axes[1, 0].plot(range(99, len(scores)), ma, linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Game Scores')
    axes[1, 0].grid(True, alpha=0.3)

    # Lengths
    axes[1, 1].plot(lengths, alpha=0.3)
    if len(lengths) >= 100:
        ma = np.convolve(lengths, np.ones(100)/100, mode='valid')
        axes[1, 1].plot(range(99, len(lengths)), ma, linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Moves')
    axes[1, 1].set_title('Episode Lengths')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f'plots/parallel_training_ep{episode}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Plot saved: {out_path}")


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

    agent, rewards, max_tiles, scores = train_dqn_parallel(
        num_episodes=50000,
        num_envs=16,
        updates_per_step=4,
    )
