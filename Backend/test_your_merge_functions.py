# test_your_merge_functions.py
"""
Test if merge functions work correctly
"""
import sys
sys.path.append('Backend')
from game_2048_env import Game2048Env
import numpy as np

env = Game2048Env()

print("="*70)
print("TESTING ALL MERGE FUNCTIONS")
print("="*70)

# Test 1: UP
print("\n" + "="*70)
print("TEST 1: MERGE UP")
print("="*70)
env.board = np.array([[2, 4, 2, 4],
                      [2, 8, 4, 8],
                      [4, 16, 8, 16],
                      [8, 32, 16, 32]])
print("Before UP:")
print(env.board)

new_board, reward = env.merge_up(env.board)
print("\nAfter UP:")
print(new_board)

print("\nExpected:")
print("Column 0: [2,2,4,8] → [4,4,8,0] (2+2=4)")
print("Column 1: [4,8,16,32] → [4,8,16,32] (no merges)")
print("Column 2: [2,4,8,16] → [2,4,8,16] (no merges)")
print("Column 3: [4,8,16,32] → [4,8,16,32] (no merges)")

expected = np.array([[4, 4, 2, 4],
                     [4, 8, 4, 8],
                     [8, 16, 8, 16],
                     [0, 32, 16, 32]])
if np.array_equal(new_board, expected):
    print("\n✓ UP CORRECT")
else:
    print("\n✗ UP WRONG!")
    print("Expected:")
    print(expected)

# Test 2: DOWN
print("\n" + "="*70)
print("TEST 2: MERGE DOWN")
print("="*70)
env.board = np.array([[2, 4, 2, 4],
                      [2, 8, 4, 8],
                      [4, 16, 8, 16],
                      [8, 32, 16, 32]])
print("Before DOWN:")
print(env.board)

new_board, reward = env.merge_down(env.board)
print("\nAfter DOWN:")
print(new_board)

expected = np.array([[0, 4, 2, 4],
                     [4, 8, 4, 8],
                     [4, 16, 8, 16],
                     [8, 32, 16, 32]])
if np.array_equal(new_board, expected):
    print("\n✓ DOWN CORRECT")
else:
    print("\n✗ DOWN WRONG!")
    print("Expected:")
    print(expected)

# Test 3: LEFT
print("\n" + "="*70)
print("TEST 3: MERGE LEFT")
print("="*70)
env.board = np.array([[2, 2, 4, 8],
                      [4, 8, 16, 32],
                      [0, 0, 0, 0],
                      [2, 2, 2, 2]])
print("Before LEFT:")
print(env.board)

new_board, reward = env.merge_left(env.board)
print("\nAfter LEFT:")
print(new_board)

expected = np.array([[4, 4, 8, 0],
                     [4, 8, 16, 32],
                     [0, 0, 0, 0],
                     [4, 4, 0, 0]])
if np.array_equal(new_board, expected):
    print("\n✓ LEFT CORRECT")
else:
    print("\n✗ LEFT WRONG!")
    print("Expected:")
    print(expected)

# Test 4: RIGHT
print("\n" + "="*70)
print("TEST 4: MERGE RIGHT")
print("="*70)
env.board = np.array([[2, 2, 4, 8],
                      [4, 8, 16, 32],
                      [0, 0, 0, 0],
                      [2, 2, 2, 2]])
print("Before RIGHT:")
print(env.board)

new_board, reward = env.merge_right(env.board)
print("\nAfter RIGHT:")
print(new_board)

expected = np.array([[0, 4, 4, 8],
                     [4, 8, 16, 32],
                     [0, 0, 0, 0],
                     [0, 0, 4, 4]])
if np.array_equal(new_board, expected):
    print("\n✓ RIGHT CORRECT")
else:
    print("\n✗ RIGHT WRONG!")
    print("Expected:")
    print(expected)

# Test 5: Real game scenario
print("\n" + "="*70)
print("TEST 5: REAL GAME SCENARIO")
print("="*70)
env.board = np.array([[4, 16, 4, 0],
                      [32, 4, 8, 0],
                      [128, 2, 4, 0],
                      [16, 8, 2, 0]])
print("Board:")
print(env.board)

print("\nTesting each direction:")
for direction, name in [(0, "UP"), (1, "DOWN"), (2, "LEFT"), (3, "RIGHT")]:
    env.board = np.array([[4, 16, 4, 0],
                          [32, 4, 8, 0],
                          [128, 2, 4, 0],
                          [16, 8, 2, 0]])
    moved, reward = env.move(direction, update_board=False, update_score=False)
    print(f"  {name}: moved={moved}")

print("\nExpected:")
print("  UP: moved=True (column 2 can merge: [4,8,4,2])")
print("  DOWN: moved=True")
print("  LEFT: moved=False (already left-aligned)")
print("  RIGHT: moved=True")

# Test 6: Show what the agent sees
print("\n" + "="*70)
print("TEST 6: AGENT'S PERSPECTIVE")
print("="*70)

from dqn_educational import DQNAgent
agent = DQNAgent()

env.reset()
print("Random board:")
print(env.board)

state = env.get_observation()
print(f"\nState shape: {state.shape}")

import torch
with torch.no_grad():
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    q_values = agent.policy_net(state_tensor)[0]
    print(f"\nQ-values: {q_values}")
    print(f"Best action: {q_values.argmax().item()} ({['UP', 'DOWN', 'LEFT', 'RIGHT'][q_values.argmax().item()]})")

# Check which actions are actually valid
print("\nActually valid moves:")
for direction, name in [(0, "UP"), (1, "DOWN"), (2, "LEFT"), (3, "RIGHT")]:
    board_copy = env.board.copy()
    moved, _ = env.move(direction, update_board=False, update_score=False)
    print(f"  {name}: {'VALID' if moved else 'INVALID'}")

print("\n" + "="*70)
print("TESTING COMPLETE")
print("="*70)