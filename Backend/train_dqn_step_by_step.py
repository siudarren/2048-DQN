# train_dqn_step_by_step.py
"""
Step-by-Step DQN Training for 2048
===================================

This script walks you through training a DQN agent with detailed
explanations at each step. Perfect for learning!

The training loop:
1. Reset environment
2. For each step:
   - Select action (epsilon-greedy)
   - Execute action
   - Store experience
   - Train network (if enough samples)
3. Track progress
4. Save model periodically
"""

import sys
sys.path.append('Backend')

from game_2048_env import Game2048Env
from dqn_educational import DQNAgent, print_training_summary, board_to_planes
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time
import os


def train_dqn_educational(num_episodes=5000, agent=None):
    """
    Main training loop with detailed logging.
    
    Training Phases:
    - Episodes 0-500: Heavy exploration (epsilon ~1.0 → 0.6)
    - Episodes 500-2000: Balanced (epsilon ~0.6 → 0.2)
    - Episodes 2000+: Heavy exploitation (epsilon ~0.2 → 0.01)
    """
    
    print("\n" + "="*70)
    print("🎮 DQN TRAINING FOR 2048 - EDUCATIONAL MODE")
    print("="*70)
    print("\nThis training will teach you how DQN works step-by-step.")
    print("Watch the agent learn from random moves to strategic play!\n")
    
    # === STEP 1: Setup ===
    print("📋 Step 1: Setting up environment and agent...")
    
    env = Game2048Env()
    if agent == None:
        agent = DQNAgent(
            learning_rate=1e-4,      # How fast to learn
            gamma=1,              # How much to value future rewards
            epsilon_start=1.0,       # Start with 100% exploration
            epsilon_end=0.01,        # End with 1% exploration
            epsilon_decay=0.995,     # Decay rate per episode
            batch_size=64,           # Learn from 64 experiences at a time
            target_update_freq=1000  # Update target network every 1000 steps
        )
    
    print("✅ Environment and agent ready!")
    print(f"   Action space: {env.action_space}")
    print(f"   State space: {env.observation_space.shape}")
    
    # === STEP 2: Training Metrics ===
    print("\n📊 Step 2: Initializing tracking metrics...")
    
    episode_rewards = []      # Total reward per episode
    episode_max_tiles = []    # Max tile achieved per episode
    episode_scores = []       # Game score per episode
    episode_lengths = []      # Number of moves per episode

    
    recent_rewards = deque(maxlen=100)
    recent_max_tiles = deque(maxlen=100)
    
    print("✅ Metrics initialized!")
    
    # === STEP 3: Training Loop ===
    print("\n🚀 Step 3: Starting training loop...")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        # === Episode Start ===
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        # For first episode, show what's happening
        # print("EPISODE #{} WALKTHROUGH:".format(episode + 1))
        # print("-" * 70)
        
        # === Episode Loop ===
        while not done:
            
            # 1. Get valid moves from Env
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions=valid_actions, training=True)
            next_state, reward, done, info = env.step(action)

            
            # --- Store Experience ---
            state_planes = board_to_planes(board)          # (16, 4, 4)
            next_state_planes = board_to_planes(next_board)
            agent.memory.push(state, action, reward, next_state, done)
            
            # --- Train Agent ---
            loss = agent.train_step()
            
            if episode == 0 and episode_steps < 3 and loss is not None:
                print(f"    Loss: {loss:.4f}")
            
            # --- Update State ---
            state = next_state
            episode_reward += reward
            episode_steps += 1
        
        # print(f"\n  Episode ended after {episode_steps} steps")
        # print(f"  Final score: {env.score}")
        # print(f"  Max tile: {env.board.max()}")
        # print("-" * 70 + "\n")
        
        # === Episode End ===
        max_tile = env.board.max()
        score = env.score
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_max_tiles.append(max_tile)
        episode_scores.append(score)
        episode_lengths.append(episode_steps)
        
        recent_rewards.append(episode_reward)
        recent_max_tiles.append(max_tile)
        
        # Decay epsilon
        agent.update_epsilon()
        agent.episodes += 1
        if agent.episodes % 5000 == 0:
            agent.epsilon = 0.2
        
        # === Logging ===
        if (episode + 1) % 10 == 0:
            elapsed = time.time() - start_time
            episodes_per_sec = (episode + 1) / elapsed

            avg_reward = np.mean(recent_rewards)
            avg_max_tile = np.mean(recent_max_tiles)
            avg_score = np.mean(episode_scores[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            avg_loss = np.mean(agent.losses[-100:]) if agent.losses else 0
            
            print(f"Episode {episode + 1}/{num_episodes} "
                  f"({episodes_per_sec:.1f} ep/s)")
            print(f"  📊 Performance:")
            print(f"     Reward: {episode_reward:.1f} (avg: {avg_reward:.1f})")
            print(f"     Max tile: {max_tile} (avg: {avg_max_tile:.1f})")
            print(f"     Score: {score:.0f} (avg: {avg_score:.1f})")
            print(f"     Moves: {episode_steps} (avg: {avg_length:.1f})")
            print(f"  🎯 Training:")
            print(f"     Epsilon: {agent.epsilon:.4f}")
            print(f"     Loss: {avg_loss:.4f}")
            print(f"     Buffer: {len(agent.memory):,}/{agent.memory.capacity:,}")
            print(f"  ⏱️  Time: {elapsed/60:.1f} min")
            print("-" * 70)
        
        # === Milestone Checks ===
        # Check for learning milestones
        if episode == 100:
            print("\n🎉 MILESTONE: 100 episodes completed!")
            print(f"   Average max tile: {np.mean(episode_max_tiles[-100:]):.1f}")
            print(f"   Agent should be reaching 128-256 tiles by now.")
            print("-" * 70 + "\n")
        
        if episode == 500:
            print("\n🎉 MILESTONE: 500 episodes completed!")
            print(f"   Average max tile: {np.mean(episode_max_tiles[-100:]):.1f}")
            print(f"   Agent should be reaching 512-1024 tiles now.")
            print("-" * 70 + "\n")
        
        if episode == 1000:
            print("\n🎉 MILESTONE: 1000 episodes completed!")
            print(f"   Average max tile: {np.mean(episode_max_tiles[-100:]):.1f}")
            print(f"   Agent should be reaching 1024-2048 tiles now.")
            print("-" * 70 + "\n")
        
        # === Save Model ===
        if (episode + 1) % 500 == 0:
            os.makedirs('models', exist_ok=True)
            agent.save(f'models/dqn_2048_ep{episode+1}.pt')
            plot_progress(episode_rewards, episode_max_tiles, 
                         episode_scores, episode_lengths, episode + 1)
    
    # === Training Complete ===
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Total episodes: {num_episodes}")
    print(f"Total steps: {agent.steps:,}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"\n📈 Final Performance (last 100 episodes):")
    print(f"   Average max tile: {np.mean(episode_max_tiles[-100:]):.1f}")
    print(f"   Best max tile: {max(episode_max_tiles[-100:])}")
    print(f"   Average score: {np.mean(episode_scores[-100:]):.1f}")
    print(f"   Best score: {max(episode_scores[-100:])}")
    print("="*70 + "\n")
    
    # Save final model
    agent.save('models/dqn_2048_final.pt')
    plot_progress(episode_rewards, episode_max_tiles, 
                 episode_scores, episode_lengths, num_episodes)
    
    return agent, episode_rewards, episode_max_tiles, episode_scores


def plot_progress(rewards, max_tiles, scores, lengths, episode):
    """Create detailed training progress plots"""
    
    os.makedirs('plots', exist_ok=True)
    
    fig = plt.figure(figsize=(15, 10))
    
    # Create 2x2 grid of subplots
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # === Plot 1: Rewards ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(rewards, alpha=0.3, color='blue', label='Episode Reward')
    if len(rewards) >= 100:
        moving_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
        ax1.plot(range(99, len(rewards)), moving_avg, 'r-', 
                linewidth=2, label='Moving Avg (100)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Rewards Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # === Plot 2: Max Tiles ===
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(max_tiles, alpha=0.3, color='green', label='Max Tile')
    if len(max_tiles) >= 100:
        moving_avg = np.convolve(max_tiles, np.ones(100)/100, mode='valid')
        ax2.plot(range(99, len(max_tiles)), moving_avg, 'r-', 
                linewidth=2, label='Moving Avg (100)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Max Tile Value')
    ax2.set_title('Max Tile Achieved')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # === Plot 3: Scores ===
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(scores, alpha=0.3, color='purple', label='Score')
    if len(scores) >= 100:
        moving_avg = np.convolve(scores, np.ones(100)/100, mode='valid')
        ax3.plot(range(99, len(scores)), moving_avg, 'r-', 
                linewidth=2, label='Moving Avg (100)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Game Score')
    ax3.set_title('Game Scores Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # === Plot 4: Episode Lengths ===
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(lengths, alpha=0.3, color='orange', label='Episode Length')
    if len(lengths) >= 100:
        moving_avg = np.convolve(lengths, np.ones(100)/100, mode='valid')
        ax4.plot(range(99, len(lengths)), moving_avg, 'r-', 
                linewidth=2, label='Moving Avg (100)')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Number of Moves')
    ax4.set_title('Episode Lengths (Moves per Game)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.savefig(f'plots/training_progress_ep{episode}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plot saved: plots/training_progress_ep{episode}.png")


def watch_agent_play(agent, num_games=5):
    """
    Watch the trained agent play and see its decision-making.
    """
    print("\n" + "="*70)
    print("👀 WATCHING AGENT PLAY")
    print("="*70 + "\n")
    
    env = Game2048Env()
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    for game in range(num_games):
        print(f"\n🎮 Game {game + 1}/{num_games}")
        print("-" * 70)
        
        state = env.reset()
        done = False
        moves = 0
        
        print("Initial board:")
        env.render()
        print()
        
        while not done and moves < 10:  # Show first 10 moves
            action = agent.select_action(state, training=False)
            state, reward, done, _ = env.step(action)
            moves += 1
            
            print(f"Move {moves}: {action_names[action]}")
            print(f"Reward: {reward:.2f}")
            env.render()
            print(f"Score: {env.score}, Max tile: {env.board.max()}\n")
            
            input("Press Enter for next move...")
        
        # Finish the game silently
        while not done:
            action = agent.select_action(state, training=False)
            state, reward, done, _ = env.step(action)
            moves += 1
        
        print(f"\n📊 Game {game + 1} Summary:")
        print(f"   Total moves: {moves}")
        print(f"   Final score: {env.score}")
        print(f"   Max tile: {env.board.max()}")
        print("-" * 70)


if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    agent = DQNAgent(
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1,
        epsilon_end=0.01,
        epsilon_decay=0.999,
        batch_size=64,
        target_update_freq=1000
    )
    agent.load("models/dqn_2048_ep13000.pt")
    agent.epsilon = 0.3
    
    # Train the agent
    agent, rewards, max_tiles, scores = train_dqn_educational(num_episodes=200000, agent=agent)
    
    
  