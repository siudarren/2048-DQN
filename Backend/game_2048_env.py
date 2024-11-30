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
        self.observation_space = spaces.Box(low=0, high=2**16, shape=(4, 4), dtype=np.int)
        self.score = 0
        self.reset()

    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.add_random_tile()
        return self.get_observation()

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            row, col = random.choice(empty_cells)
            self.board[row][col] = 2 if random.random() < 0.9 else 4

    def get_observation(self):
        return self.board.copy()

    def step(self, action):
        currScore = self.score
        moved = self.move(action)
        reward = self.calculate_reward(moved, currScore)
        done = self.is_game_over()
        if moved:
            self.add_random_tile()
        return self.get_observation(), reward, done, {}

    def calculate_reward(self, moved, currScore):
        # Reward can be the sum of merged tiles or the change in the board
        # For simplicity, we can return 0 or 1 for invalid/valid moves
        return self.score - currScore if moved else -1

    def move(self, direction):
        # Implement the move logic
        # direction: 0=Up, 1=Down, 2=Left, 3=Right
        board_copy = self.board.copy()
        if direction == 0:
            self.board = self.merge_up(self.board)
        elif direction == 1:
            self.board = self.merge_down(self.board)
        elif direction == 2:
            self.board = self.merge_left(self.board)
        elif direction == 3:
            self.board = self.merge_right(self.board)
        moved = not np.array_equal(board_copy, self.board)
        return moved

    # Implement merge functions for each direction
    def merge_up(self, board):
        # Logic to merge up
        return self.merge(board)

    def merge_down(self, board):
        # Logic to merge down
        return np.flipud(self.merge(np.flipud(board)))

    def merge_left(self, board):
        # Logic to merge left
        return self.merge(board.T).T

    def merge_right(self, board):
        # Logic to merge right
        return np.fliplr(self.merge(np.fliplr(board.T)).T)

    def merge(self, board):
        new_board = np.zeros((4, 4), dtype=int)
        for i in range(4):
            tiles = board[i][board[i] != 0]
            merged_tiles = []
            skip = False
            for j in range(len(tiles)):
                if skip:
                    skip = False
                    continue
                if j + 1 < len(tiles) and tiles[j] == tiles[j + 1]:
                    self.score += tiles[j] * 2
                    merged_tiles.append(tiles[j] * 2)
                    skip = True
                else:
                    merged_tiles.append(tiles[j])
            new_board[i, :len(merged_tiles)] = merged_tiles
        return new_board

    def is_game_over(self):
        if np.any(self.board == 0):
            return False
        # Check if any moves are possible
        for direction in range(4):
            board_copy = self.board.copy()
            moved = self.move(direction)
            self.board = board_copy  # Reset board
            if moved:
                return False
        return True

    def render(self, mode='human'):
        print(self.board)
