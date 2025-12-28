# backend/game_2048_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from .dqn_educational import board_to_planes

class Game2048Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Game2048Env, self).__init__()
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=1, shape=(16, 4, 4), dtype=np.float32)
        self.score = 0
        self.reset()

    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()  # Add a second random tile
        # print("Environment: Resetting the game with two tiles.")
        return self.get_observation()

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            row, col = random.choice(empty_cells)
            self.board[row][col] = 2 if random.random() < 0.9 else 4
            # print(f"Environment: Added tile {self.board[row][col]} at position ({row}, {col}).")

    def get_observation(self):
        # Normalize using log2, handle zero tiles
        planes = board_to_planes(self.board)      # (16, 4, 4), float32
        return planes

    def step(self, action):
        prev_max_tile = np.max(self.board)
        prev_score = self.score
        moved, merge_reward = self.move(action)
        done = False

        if moved:
            # === VALID MOVE ===
            # print("Valid Move")
            # Base reward from merging
            reward = self.score - prev_score
            
            # Add random tile after valid move
            self.add_random_tile()

            # # Bonus for achieving higher tiles
            # new_max_tile = np.max(self.board)
            # if new_max_tile > prev_max_tile:
            #     reward += np.log2(new_max_tile) * 2
            
            # # Small bonus for keeping board open (encourages not filling up)
            # empty_cells = np.sum(self.board == 0)
            # reward += empty_cells * 0.1

            # CHECK IF GAME IS OVER (this was missing!)
            done = self.is_game_over()
            
        else:
            # === INVALID MOVE ===
            # Strong penalty to discourage invalid moves
            # print("Invalid Move")
            reward = 0
        
        
        
        # Additional penalty if game over (ran out of moves)
        if done:
            # reward -= 50.0
            pass
        
        return self.get_observation(), reward, done, {}




    def get_additional_reward(self):
        empty_cells = np.sum(self.board == 0)
        return empty_cells * 0.1  # Small reward for having more empty cells

    def move(self, direction, update_score=True, update_board=True):
        # Always work with a copy to avoid reference issues
        board_to_process = self.board.copy()
        
        if direction == 0:
            new_board, reward = self.merge_up(board_to_process)
        elif direction == 1:
            new_board, reward = self.merge_down(board_to_process)
        elif direction == 2:
            new_board, reward = self.merge_left(board_to_process)
        elif direction == 3:
            new_board, reward = self.merge_right(board_to_process)
        else:
            raise ValueError("Invalid action.")

        # Check if board actually changed
        moved = not np.array_equal(self.board, new_board)

        if moved and update_board:
            self.board = new_board
        
        if moved and update_score:
            self.score += reward

        return moved, reward
        


    # Merge functions return new board and reward obtained from merges
    def merge_up(self, board):
        transposed = board.T
        merged_board, reward = self.merge(transposed)
        result = merged_board.T
        return result, reward

    def merge_down(self, board):
        flipped = np.flipud(board)       # flip vertically
        transposed = flipped.T           # columns → rows
        merged_board, reward = self.merge(transposed)  # merge left
        untransposed = merged_board.T
        result = np.flipud(untransposed) # flip back vertically
        return result, reward

    def merge_left(self, board):
        merged_board, reward = self.merge(board)
        return merged_board, reward

    def merge_right(self, board):
        flipped = np.fliplr(board)
        merged_board, reward = self.merge(flipped)
        result = np.fliplr(merged_board)
        return result, reward

    def merge(self, board):
        new_board = np.zeros((4, 4), dtype=int)
        reward = 0
        for i in range(4):
            # Extract the current row
            row = board[i]
            # Extract non-zero tiles
            non_zero_tiles = [tile for tile in row if tile != 0]
            # Merge tiles
            merged_tiles = []
            index = 0
            while index < len(non_zero_tiles):
                if index + 1 < len(non_zero_tiles) and non_zero_tiles[index] == non_zero_tiles[index + 1]:
                    # Merge the tiles
                    merged_value = non_zero_tiles[index] * 2
                    merged_tiles.append(merged_value)
                    reward += merged_value
                    # print(f"Environment: Tile merged: {non_zero_tiles[index]} + {non_zero_tiles[index+1]} = {merged_value}. Reward += {merged_value}")
                    index += 2  # Skip the next tile since it's merged
                else:
                    merged_tiles.append(non_zero_tiles[index])
                    index += 1
            # Fill the remaining spaces with zeros
            merged_tiles.extend([0] * (4 - len(merged_tiles)))
            # Check if this new row differs from the original row (for debugging)
            if not np.array_equal(merged_tiles, row):
                # print(f"Environment: Row {i} after merge: {merged_tiles}")
                pass
            # Place the merged row back into the new_board
            new_board[i] = merged_tiles
        return new_board, reward

    def get_valid_actions(self):
            """Returns list of valid actions [0, 1, 2, 3]"""
            valid_actions = []
            # Check all 4 directions
            for action in range(4):
                # Test move on a copy of the board
                # We reuse your existing move() logic but suppress updates
                moved, _ = self.move(action, update_score=False, update_board=False)
                if moved:
                    valid_actions.append(action)
            return valid_actions

    def is_game_over(self):
        # First check: any empty cells?
        if np.any(self.board == 0):
            return False
        
        # Second check: any possible merges?
        # Check horizontally
        for i in range(4):
            for j in range(3):
                if self.board[i][j] == self.board[i][j+1]:
                    return False
        
        # Check vertically
        for i in range(3):
            for j in range(4):
                if self.board[i][j] == self.board[i+1][j]:
                    return False
        
        # No empty cells and no possible merges
        return True



    def render(self, mode='human'):
        print("Current Board:")
        print(self.board)
