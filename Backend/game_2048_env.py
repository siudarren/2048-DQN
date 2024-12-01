# backend/game_2048_env.py

import gym
from gym import spaces
import numpy as np
import random

class Game2048Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Game2048Env, self).__init__()
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=2**16, shape=(4, 4, 1), dtype=np.float32)
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
        observation = np.log2(self.board + 1) / np.log2(65536)
        observation = observation.reshape(4, 4, 1).astype(np.float32)
        return observation

    def step(self, action):
        moved, reward = self.move(action, update_score=True, update_board=True)
        done = self.is_game_over()
        if moved:
            self.add_random_tile()
            reward += self.get_additional_reward()
        else:
            reward = 0
        return self.get_observation(), reward, done, {}


    def get_additional_reward(self):
        empty_cells = np.sum(self.board == 0)
        return empty_cells * 0.1  # Small reward for having more empty cells

    def move(self, direction, update_score=True, update_board=True):
        board_copy = self.board.copy()
        if direction == 0:
            new_board, reward = self.merge_up(self.board)
        elif direction == 1:
            new_board, reward = self.merge_down(self.board)
        elif direction == 2:
            new_board, reward = self.merge_left(self.board)
        elif direction == 3:
            new_board, reward = self.merge_right(self.board)
        else:
            raise ValueError("Invalid action.")

        moved = not np.array_equal(board_copy, new_board)

        if moved:
            if update_board:
                self.board = new_board
            if update_score:
                self.score += reward
                # print(f"Environment: Moved in direction {direction}. Score increased by {reward} to {self.score}.")
        else:
            # print(f"Environment: Move in direction {direction} did not change the board.")
            pass

        return moved, reward


    # Merge functions return new board and reward obtained from merges
    def merge_up(self, board):
        merged_board, reward = self.merge(board)
        return merged_board, reward

    def merge_down(self, board):
        rotated_right1 = np.rot90(board, k=1)
        rotated_right2 = np.rot90(rotated_right1, k=1)

        merged_board, reward = self.merge(rotated_right2)

        rotate_left1 = np.rot90(merged_board, k=-1)
        rotate_left2 = np.rot90(rotate_left1, k=-1)
        return rotate_left2, reward

    def merge_left(self, board):
        rotated_left = np.rot90(board, k=1)
        merged_board, reward = self.merge(rotated_left)
        rotated_right = np.rot90(merged_board, k=-1)
        return rotated_right, reward

    def merge_right(self, board):
        rotated_right = np.rot90(board, k=-1)
        merged_board, reward = self.merge(rotated_right)
        rotated_left = np.rot90(merged_board, k=1)
        return rotated_left, reward

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
                    reward += np.log2(merged_value)
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


    def is_game_over(self):
        if np.any(self.board == 0):
            # print("Environment: Not game over - there are empty cells.")
            return False
        for direction in range(4):
            moved, _ = self.move(direction, update_score=False, update_board=False)
            if moved:
                # print(f"Environment: Game not over - move possible in direction {direction}.")
                return False
        # print("Environment: Game over - no moves left.")
        return True



    def render(self, mode='human'):
        print("Current Board:")
        print(self.board)
