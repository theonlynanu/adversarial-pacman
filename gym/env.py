import gymnasium as gym
from gymnasium import spaces
import numpy as np

from pacman_engine.pacman import Game, Directions, ClassicGameRules

class PacmanEnv(gym.Env):
    metadata = {"render_modes": ["pretty", "text"]}
    
    def __init__(self, layout="mediumClassic", render_mode="pretty"):
        super().__init__()
        self.layout = layout
        self.render_mode = render_mode
        self._actions = Directions
        self.action_space = spaces.Discrete(4)
        
        self.observation_space = spaces.Box(
            low = 0, high = 255, shape = (84, 84, 3), dtype=np.uint8
        )
        
        self._start_game()
        
    def _start_game(self):
        self.game = Game("keyboardagent", ["randomghost", 1], ClassicGameRules)
        
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._start_game()
        frame = self._render_frame()
        return frame, {}
    
    
    def step(self, action):
        dir_str = self._actions[action]
        reward = self.game.move_pacman(dir_str)
        done = self.game.game_over
        frame = self._render_frame()
        return frame, reward, done, False, {}
    
    
    def _render_frame(self):
        img = self.game.
        