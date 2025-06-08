import gymnasium as gym
from gymnasium import spaces
import numpy as np

from pacman_engine.pacman import Directions, ClassicGameRules, GameState
from pacman_engine.game import Game, GameStateData, Actions
from pacman_engine.layout import Layout, get_layout



""" -------------------- CONSTANTS AND HELPERS -------------------- """
DIRECTIONS_TO_IDX = {
    'North': 0,
    'South': 1,
    'East': 2,
    'West': 3,
    "Northeast": 4,
    "Northwest": 5,
    "Southeast": 6,
    "Southwest": 7
}



""" ========================== MAIN CLASS ========================== """
class PacmanEnv(gym.Env):
    """ Gymnasium wrapper around the Berkeley Pac-Man GameState object.
    
    Parameters:
        layout:     str - Layout file name (default: "mediumClassic")
        num_ghosts: int - Maximum number of ghosts that can be spawned from the
                    layout (default: 4)
        obs_type:   str - Type of observations the agent can make. 'grid' for a
                    full board view, 'directional' for an immediate view + directional
                    view of the ghosts (default: 'grid')
        training_agent: str - 'pacman' | 'ghost', defines the agent currently being trained
        ghost_train_idx: int - which ghost agent is being trained, if training_agent == 'ghost'
        
                    
        
        TODO -  Add selection of which agent is being trained, and default behaviors for non-training agents to
        TODO -  allow use of trained policy 
        # ghost_agents: List[str]
        # pacman_agent: str
        """
    metadata = {"render_modes": ["graphics", "text"]}
    
    def __init__(self, *, 
                 layout=        "mediumClassic",
                 render_mode=   "graphics",
                 num_ghosts=    4,
                 obs_type=      "grid",
                 training_agent=   "pacman",
                 ghost_train_idx=   1,
                 ):
        
        super().__init__()
        
        self.layout_name = layout
        self.render_mode = render_mode
        self.num_ghosts = min(max(1, num_ghosts), 4)    # Confine num_ghosts to [1,4]
        self.obs_type = obs_type
        self.training_agent = training_agent
        self.ghost_train_idx = ghost_train_idx
        
        # Action space 
        self._actions = [
            Directions.NORTH,
            Directions.SOUTH,
            Directions.EAST,
            Directions.WEST,
            Directions.STOP     # Not sure if we want to allow stopping, but it's
                                # in the original spec so we can include it for now
        ]
        self.action_space = spaces.Discrete(len(self._actions))
        
        
        
        self.reset()
        
    """ _________ GYM API _________ """
    def _reset_game(self):
        self.state = GameState()
        layout = get_layout(self.layout_name)
        self.state.initialize(layout, self.num_ghosts)
    

    def reset(self, *, seed= None, options: None):
        super().reset(seed=seed)
        self._reset_game()
        return self._make_obs(), {}
    
    def step(self, action: int):
        pass
            
    
    # TODO - Extend this to be able to render in-terminal using text_display.py
    def render(self):
        if self.render_mode != "graphics":
            return None
    

    def close(self):
        pass
        
        
    """ ~~~~~~~~~~~ Observation Helpers ~~~~~~~~~~~ """
        
    def _build_observation_space(self):
        layout = get_layout(self.layout_name)
        height, width = layout.height, layout.width
        if self.obs_type == "grid":
            # wall, pellet, power pellet, pacman, ghost, scared ghost
            self.observation_space = spaces.Box(low=0, high=1, shape=(height, width, 6))
        else:
            raise ValueError(f"Unknown obs_type: {self.obs_type}, only option right now is 'grid'")
        
    def _make_obs(self):
        if self.obs_type == 'grid':
            return self._obs_grid()
        else:
            raise ValueError(f"Unknown obs_type: {self.obs_type}, only option right now is 'grid'")
            
        
        
    def _obs_grid(self):
        data = self.state.data
        h, w = data.layout.height, data.layout.width
        grid = np.zeros((h, w, 6), dtype=int)
        
        # Check walls - we want grid[x, y, 0] = data.layout.walls[x][y]
        grid[:, :, 0] = np.transpose(int(data.layout.walls))
        
        # Check pellets
        grid[:, :, 1] = np.transpose(int(data.layout.food))
        
        # Check power pellets
        grid[:, :, 2] = np.transpose(int(data.capsules))
        
        # Check pacman
        x, y = self.state.get_pacman_position()
        grid[int(y), int(x), 3] = 1
        
        # Check ghost(s) including scared state
        for ghost in self.state.get_ghost_states():
            x, y = ghost.get_position()
            layer = 5 if ghost.scared_timer > 0 else 4
            grid[int(y), int(x), layer] = 1
            
        return grid
        

        
            
            
        