import gymnasium as gym
from gymnasium import spaces
import numpy as np

from pacman_engine.pacman import Directions, GameState
from pacman_engine.layout import get_layout
from pacman_engine.keyboard_agents import KeyboardAgent
from pacman_engine.ghost_agents import RandomGhost
from pacman_engine.graphics_utils import begin_graphics

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

#Constants for reward calculations
REWARD_PELLET = 1
REWARD_POWER = 10
REWARD_GHOST = 50
REWARD_DEATH = -500
REWARD_STEP = -1



""" ========================== MAIN CLASS ========================== """
class PacmanEnv(gym.Env):
    """ Gymnasium wrapper around the Berkeley Pac-Man GameState object.
    
    Parameters:
        layout:     str - Layout file name (default: "mediumClassic")
        render_mode:str - how the game should be rendered, if at all. Meant to extend the
                    capabilities of the base game engine to either create the GUI window
                    or run the text game in-terminal. (default: 'graphics')
        num_ghosts: int - Maximum number of ghosts that can be spawned from the
                    layout (default: 4)
        obs_type:   str - Type of observations the agent can make. 'grid' for a
                    full board view, 'directional' for an immediate view + directional
                    view of the ghosts (default: 'grid')
        training_agent: str - 'pacman' | 'ghost' | None, defines the agent currently being trained
        ghost_train_idx: int - which ghost agent is being trained, if training_agent == 'ghost'
        pacman_agent:   str - defines the agent that should play for pacman
        ghost_agents    List[str] | None - list of agents that play for each ghost
        """
    metadata = {"render_modes": ["graphics", "text", None],
                }
    
    def __init__(self,
                 layout=        "mediumClassic",
                 render_mode=   "graphics",
                 obs_type=      "grid",
                 training_agent=   None,
                 ghost_train_idx=   None,
                 pacman_agent = None,
                 ghost_agents = None
                 ):
        
        super().__init__()
        
        # Retrieve layout object
        self.layout_name = layout
        self.layout = get_layout(self.layout_name)
        
        self.render_mode = render_mode
        
        # This is gonna be changeable later, once we get our basic loop working
        self.obs_type = obs_type
        
        # Setup up training parameters
        self.training_mode = training_agent         # This way 'None' communicates no training to be done
        self.ghost_train_idx = ghost_train_idx
        self.num_ghosts = self.layout.get_num_ghosts()   # num_ghosts now automatically retrieved
        
        begin_graphics()
        
        # Attach agent objects if provided
        self.pacman_agent = pacman_agent or KeyboardAgent()
        if ghost_agents is None:
            ghost_agents = [RandomGhost(i + 1) for i in range(self.num_ghosts)]
        if len(ghost_agents) < self.num_ghosts:
            raise ValueError("Not enough ghost Agents were provided for this layout")
        self.ghost_agents = ghost_agents
        
        # Intialize GameState object - removed since reset() will do it anyway
        # self.state = GameState()
        # self.state.initialize(layout, self.num_ghosts)
        
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
        
        self._build_observation_space()
        
        self.reset()
        
        
        
    """ _________ GYM API _________ """

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.layout = get_layout(self.layout_name)
        self.num_ghosts = self.layout.get_num_ghosts()
        
        self.state = GameState()
        self.state.initialize(self.layout, self.num_ghosts)
        self.prev_score = self.state.get_score()
        
        return self._make_obs(), {}
    
    def step(self, action: int):
        # Determine pacman action
        if self.training_mode == 'pacman':
            # Apply action to Pacman
            pacman_action = self._actions[action]
            if pacman_action not in self.state.get_legal_actions(0):    # Placeholder for now, this should be
                print("Illegal move attempted for pacman, making random legal move.")
                pacman_action = self._random_legal(0)                   # unnecessary if we get the rest right
        else:
            pacman_action = self.pacman_agent.get_action(self.state)
            
        self.state = self.state.generate_pacman_successor(pacman_action)
        if self.render_mode == 'graphics': self.render()

        terminated = self.state.is_win() or self.state.is_lose()


            # DA - Removed this previous loop due to repetitiveness - kept this here to see our first draft
            # if not self.state.is_win() and not self.state.is_lose():
            #     # Default ghost behaviors
            #     for ghost_idx in range(1, self.state.get_num_agents()):
            #         ghost_action = self._random_legal(ghost_idx)  # Replace with smarter logic later
            #         self.state = self.state.generate_successor(ghost_idx, ghost_action)

        # Determine ghost actions
        if self.training_mode == 'ghost' and (  # Quick little sanity check. It's late and I forgot 
            self.ghost_train_idx is None or     # where all this is already checked - might be unnecessary
            self.ghost_train_idx < 1 or
            self.ghost_train_idx > self.num_ghosts
        ):
            raise ValueError("Invalid ghost_train_idx for this layout")
        for ghost_idx in range(1, self.state.get_num_agents()):
            # Exit early if game is already over
            if terminated:
                break
            if self.training_mode == 'ghost' and ghost_idx == self.ghost_train_idx:
                ghost_action = self._actions[action]
                if ghost_action not in self.state.get_legal_actions(ghost_idx):
                    print("Illegal move attempted for ghost, making random legal move.")
                    ghost_action = self._random_legal(ghost_idx)
            else:
                ghost_action = self.ghost_agents[ghost_idx - 1].get_action(self.state)
            
            self.state = self.state.generate_successor(ghost_idx, ghost_action)
            self.render()
            terminated = self.state.is_lose() or self.state.is_win()
            
            
            
        
        # elif self.training_mode == 'ghost':
        #     # Apply action to ghost being trained
        #     ghost_action = self._actions[action]
        #     self.state = self.state.generate_successor(self.ghost_train_idx, ghost_action)
        #     if not self.state.is_win() and not self.state.is_lose():

        #         # Default behavior for Pacman
        #         pacman_action = Directions.STOP  # or a simple rule-based policy
        #         self.state = self.state.generate_successor(0, pacman_action)

        #         # Default for other ghosts
        #         for ghost_idx in range(1, self.state.get_num_agents()):
        #             if ghost_idx == self.ghost_train_idx:
        #                 continue
        #             self.state = self.state.generate_successor(ghost_idx, Directions.STOP)
        # else:
        #     raise ValueError(f"Unknown training_mode: {self.training_mode}")

        # Compute reward and done
        reward = self.state.get_score() - self.prev_score
        self.prev_score = self.state.get_score()
        truncated = False # Keep this here in case we decide to set time limits
        info = {"score": self.state.get_score()}

        return self._make_obs(), reward, terminated, truncated, info


    # TODO - Extend this to be able to render in-terminal using text_display.py
    def render(self):
        if self.render_mode == "graphics":
            self._ensure_display()
            self._display.update(self.state.data)
        elif self.render_mode == "text":
            # Placeholder for future terminal rendering support
            print(self.state)
        elif self.render_mode is None:  # Allows for silent running
            return
        else:
            raise ValueError(f"Unsupported render_mode: {self.render_mode}")
        
        
    def close(self):
        if hasattr(self, "_display"):
            self._display.finish()
            del self._display
        
        
    ''' ################# GENERAL HELPERS ################# '''
        
    def _random_legal(self, agent_idx: int):
        legal = self.state.get_legal_actions(agent_idx)
        if legal:
            return self.np_random.choice(legal) # Updated to use Gym's seeded random choice
        else:
            return self.state.data.agent_states[agent_idx].get_direction()
        
        
    def _ensure_display(self):
        # Ensures that a _display delegate object exists if needed
        if self.render_mode != 'graphics':
            return
        if hasattr(self, "_display"):
            return
        from pacman_engine.graphics_display import PacmanGraphics
        self._display = PacmanGraphics()
        self._display.initialize(self.state.data)
            
        
        
    """ ~~~~~~~~~~~ Observation Helpers ~~~~~~~~~~~ """
        
    def _build_observation_space(self):
        height, width = self.layout.height, self.layout.width
        if self.obs_type == "grid":
            # wall, pellet, power pellet, pacman, ghost, scared ghost
            self.observation_space = spaces.Box(low=0, high=1, shape=(height, width, 6), dtype=int)
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
        
        # Check walls and food - we want grid[x, y, 0] = data.layout.walls[x][y]
        for x in range(w):
            for y in range(h):
                if data.layout.walls[x][y]:
                    grid[y, x, 0] = 1
                if data.food[x][y]:
                    grid[y, x, 1] = 1
        
        # Check power pellets
        for cx, cy in data.capsules:
            grid[cy, cx, 2] = 1
        
        # Check pacman
        x, y = self.state.get_pacman_position()
        grid[int(y), int(x), 3] = 1
        
        # Check ghost(s) including scared state
        for ghost in self.state.get_ghost_states():
            x, y = ghost.get_position()
            layer = 5 if ghost.scared_timer > 0 else 4
            grid[int(y), int(x), layer] = 1
            
        return grid
        