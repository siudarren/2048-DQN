# test_game_over_comprehensive.py
"""
Comprehensive test to find the bug in is_game_over()
"""
import sys
sys.path.append('Backend')
from game_2048_env import Game2048Env
import numpy as np

print("="*70)
print("COMPREHENSIVE GAME OVER TESTING")
print("="*70)

env = Game2048Env()

# Test 1: Board with empty cells
print("\n1. Board with empty cells (should NOT be game over)")
env.board = np.array([[16, 128, 256, 64],
                      [8, 16, 32, 8],
                      [2, 8, 2, 4],
                      [0, 0, 0, 0]])
print(env.board)
result = env.is_game_over()
print(f"is_game_over(): {result}")
print(f"Expected: False")
print(f"✓ PASS" if result == False else "✗ FAIL")

# Test 2: Full board with horizontal merge possible
print("\n2. Full board with horizontal merge (should NOT be game over)")
env.board = np.array([[2, 2, 4, 8],
                      [4, 8, 2, 4],
                      [2, 4, 8, 2],
                      [4, 2, 4, 8]])
print(env.board)
result = env.is_game_over()
print(f"is_game_over(): {result}")
print(f"Expected: False (2,2 in row 0 can merge)")
print(f"✓ PASS" if result == False else "✗ FAIL")

# Test 3: Full board with vertical merge possible
print("\n3. Full board with vertical merge (should NOT be game over)")
env.board = np.array([[2, 4, 8, 16],
                      [2, 8, 4, 2],
                      [4, 2, 8, 4],
                      [8, 4, 2, 8]])
print(env.board)
result = env.is_game_over()
print(f"is_game_over(): {result}")
print(f"Expected: False (2,2 in column 0 can merge)")
print(f"✓ PASS" if result == False else "✗ FAIL")

# Test 4: Actually game over - no moves
print("\n4. Full board, NO merges possible (SHOULD be game over)")
env.board = np.array([[2, 4, 2, 4],
                      [4, 2, 4, 2],
                      [2, 4, 2, 4],
                      [4, 2, 4, 2]])
print(env.board)
result = env.is_game_over()
print(f"is_game_over(): {result}")
print(f"Expected: True")
print(f"✓ PASS" if result == True else "✗ FAIL")

# Test 5: Test all 4 move directions
print("\n5. Testing move validity for game over board")
env.board = np.array([[2, 4, 2, 4],
                      [4, 2, 4, 2],
                      [2, 4, 2, 4],
                      [4, 2, 4, 2]])
print(env.board)
print("Testing each direction:")
for direction, name in enumerate(['UP', 'DOWN', 'LEFT', 'RIGHT']):
    moved, _ = env.move(direction, update_score=False, update_board=False)
    print(f"  {name}: moved={moved} (should be False)")

# Test 6: Play random game and track invalid moves
print("\n" + "="*70)
print("6. Playing random game and tracking behavior")
print("="*70)

env.reset()
invalid_count = 0
valid_count = 0
step = 0
max_steps = 1000

print(f"Starting board:")
print(env.board)

while step < max_steps:
    action = np.random.randint(0, 4)
    
    # Check what will happen
    moved, _ = env.move(action, update_score=False, update_board=False)
    
    # Actually do it
    obs, reward, done, _ = env.step(action)
    step += 1
    
    if moved:
        valid_count += 1
    else:
        invalid_count += 1
        # Track consecutive invalid moves
        if invalid_count > valid_count and step > 10:
            print(f"\n⚠️  WARNING at step {step}:")
            print(f"   Invalid moves: {invalid_count}")
            print(f"   Valid moves: {valid_count}")
            print(f"   Board state:")
            print(env.board)
            print(f"   is_game_over(): {env.is_game_over()}")
            
            # Test if ANY move is valid
            any_valid = False
            for test_action in range(4):
                test_moved, _ = env.move(test_action, update_score=False, update_board=False)
                if test_moved:
                    any_valid = True
                    print(f"   Action {test_action} is VALID!")
            
            if not any_valid:
                print(f"   NO valid moves - should be game over!")
                if not env.is_game_over():
                    print("   ✗✗✗ BUG: is_game_over() returned False but no moves valid!")
                    break
    
    if done:
        print(f"\n✓ Game ended normally after {step} steps")
        print(f"   Valid moves: {valid_count}")
        print(f"   Invalid moves: {invalid_count}")
        print(f"   Final board:")
        print(env.board)
        break
    
    # Print progress
    if step % 100 == 0:
        print(f"  Step {step}: Max tile={env.board.max()}, Valid={valid_count}, Invalid={invalid_count}")

if step >= max_steps:
    print(f"\n✗✗✗ GAME STUCK - reached {max_steps} steps without ending!")
    print(f"Final board:")
    print(env.board)
    print(f"is_game_over(): {env.is_game_over()}")
    print(f"Valid moves: {valid_count}, Invalid moves: {invalid_count}")
    
    # Check each direction
    print("\nTesting each direction:")
    for direction, name in enumerate(['UP', 'DOWN', 'LEFT', 'RIGHT']):
        moved, _ = env.move(direction, update_score=False, update_board=False)
        print(f"  {name}: moved={moved}")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)