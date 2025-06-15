import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time

from pacman_engine.pacman import Directions, GameState
from pacman_engine.layout import get_layout
from pacman_engine.keyboard_agents import KeyboardAgent
from pacman_engine.ghost_agents import RandomGhost
from pacman_engine.graphics_utils import begin_graphics
from pacman_engine.util import manhattan_distance



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
REWARD_PELLET = 1   # Eating a pellet
REWARD_POWER = 10   # Eating a power pellet
REWARD_GHOST = 50   # Eating a ghost
REWARD_WIN = 500
REWARD_DEATH = -500 # Dying.
REWARD_STEP = -0.1    # Taking a step without eating a pellet - currently unused since I think a general delay might work
REWARD_DELAY = -0.05 # Time delay to incentivize fast play. Might need to remove if
                    # this messes up how ghosts try to minimize scoring
from pacman_engine.pacman import SCARED_TIME
MAX_GHOST_SCARED_TIME = SCARED_TIME

""" ========================== MAIN CLASS ========================== """
class PacmanEnv(gym.Env):
    """ Gymnasium wrapper around the Berkeley Pac-Man GameState object.
    
    Parameters:
        layout:     str - Layout file name (default: "mediumClassic")
        render_mode:str - how the game should be rendered, if at all. Meant to extend the
                    capabilities of the base game engine to either create the GUI window
                    or run the text game in-terminal. (default: 'graphics')
        obs_type:   str - Type of observations the agent can make. 'grid' for a
                    full board view, plan to add 'directional' for partially observable, immediate view
                    + directional view of the ghosts (default: 'grid')
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
                 ghost_agents = None,
                 frame_time = 0.05,     # 20fps, default of the base game as well
                 ):
        
        super().__init__()
        
        # Retrieve layout object
        self.layout_name = layout
        self.layout = get_layout(self.layout_name)
        
        self.render_mode = render_mode
        self.frame_time = frame_time
        
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
            from pacman_engine.ghost_agents import AStarGhost
            shared_info = {}
            ghost_agents = [
                AStarGhost(1, shared_info),
                AStarGhost(2, shared_info),
                *[RandomGhost(i + 1) for i in range(2, self.num_ghosts)]
            ]
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
        
        self.episode_reward = 0.0
        
        return self._make_obs(), {}
    
    def step(self, action: int):
        before = self._get_snapshot()
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

        # Compute reward and done
        after = self._get_snapshot()
        reward = self._reward_from_snapshot(before, after)
        ghost_dist = self._get_min_ghost_distance()  # Add exponential penalty based on ghost distance (closer = higher penalty)
        ghost_weight = np.exp(-0.5 * ghost_dist)   # decay factor can be tuned
        reward -= ghost_weight * 5                 # scale can be tuned too
        self.episode_reward += reward       # Not sure if this is strictly necessary, but it feels useful
        truncated = False # Keep this here in case we decide to set time limits
        info = {"calculated reward": reward, "native score": self.state.get_score(), "cumulative reward": self.episode_reward}

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
        self._display = PacmanGraphics(frame_time=self.frame_time)
        self._display.initialize(self.state.data)
        
    def _get_snapshot(self):
        """Return a tuple of state features (not exact state representation)
        that can inform our scoring. Does NOT encode state itself - we'll need
        a full hash of the observation to do that."""
        
        data = self.state.data
        eaten_ghosts = sum(data._eaten[1:])
        return (
            data.food.count(),  # Pellets remaining
            len(data.capsules), # Power pellets remaining
            eaten_ghosts,
            self.state.is_win(),# In win state?
            self.state.is_lose()# In lose state?
        )
        
    def _reward_from_snapshot(self, before, after):
        b_pellets, b_power, b_eaten, b_win, b_lose = before
        a_pellets, a_power, a_eaten, a_win, a_lose = after
        
        reward = 0.0
        
        reward += REWARD_PELLET * (b_pellets - a_pellets)    # Determine difference in pellets
        reward += REWARD_POWER * (b_power - a_power)         # and power pellets
        reward += REWARD_GHOST * (b_eaten - a_eaten)         # Was a ghost eaten? (I'm not 100 sure this actually works)
        reward += REWARD_DELAY                               # Stable per-snapshot delay
        
        if a_win and not b_win:
            reward += REWARD_WIN
        if a_lose and not b_lose:
            reward += REWARD_DEATH
            
        return reward
    
    #Add discounted/exponential reward shaping based on ghost distance
    def _get_min_ghost_distance(self):
        pac_pos = self.state.get_pacman_position()
        ghost_positions = [g.get_position() for g in self.state.get_ghost_states()]
        distances = [manhattan_distance(pac_pos, ghost) for ghost in ghost_positions]
        return min(distances) if distances else float('inf')


            
        
        
    """ ~~~~~~~~~~~ Observation Helpers ~~~~~~~~~~~ """
        
    def _build_observation_space(self):
        height, width = self.layout.height, self.layout.width
        if self.obs_type == "grid":
            # wall, pellet, power pellet, pacman, ghost, scared ghost
            self.observation_space = spaces.Box(low=0, high=1, shape=(height, width, 6), dtype=int)
        elif self.obs_type == 'condensed_grid':
            """
            Our final condensed_grid output should look like:
            {
                walls: [h x w matrix with one-hot values for wall locations]
                pellets: [h x w matrix with one-hot values for pellet locations]
                pacman: (y, x) or (row, col)
                ghosts: array of (row, col, scared_timer) arrays
                power_pellets: array of (row, col) coordinate arrays
            }
            """
            coord = spaces.Box(
                low = np.array([0,0]),
                high = np.array([height - 1, width - 1]),
                shape=(2,),
                dtype=np.int8
            )
            
            ghost = spaces.Box(low = np.array([0,0,0], dtype=np.int8),
                               high = np.array([height - 1, width - 1, MAX_GHOST_SCARED_TIME], dtype=np.int8),
                               dtype=np.int8
                               )
            
            self.observation_space = spaces.Dict({
                "walls": spaces.Box(0, 1, shape=(height, width), dtype=np.int8),
                "pellets": spaces.Box(0, 1, shape=(height, width), dtype=np.int8),
                "pacman": coord,
                "ghosts": spaces.Sequence(ghost),
                "power_pellets": spaces.Sequence(coord)
            })
            
        else:
            raise ValueError(f"Unknown obs_type: {self.obs_type}, only option right now is 'grid'")
        
    def _make_obs(self):
        if self.obs_type == 'grid':
            return self._obs_grid()
        elif self.obs_type == 'condensed_grid':
            return self._obs_condensed()
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
    
    def _obs_condensed(self):
        data = self.state.data
        
        walls = np.asarray(data.layout.walls.data, dtype=np.int8).T  # Shape (H, W)
        pellets = np.asarray(data.food.data, dtype=np.int8).T        # Shape (H, W)
        
        # Swap x and y to account for row-major operations
        px, py = map(int, self.state.get_pacman_position())
        pacman = np.array([py, px], dtype=np.int8)              # (row, col)
        
        ghosts = []
        for ghost in self.state.get_ghost_states():
            # Same as above, swapping to keep the output row-major
            gx, gy = map(int, ghost.get_position())
            timer = int(ghost.scared_timer)
            ghosts.append(np.array([gy, gx, timer], dtype=np.int8))
            
        power_pellets = [np.array([cy, cx], dtype=np.int8) for (cx, cy) in data.capsules]
        
        return {
            "walls": walls,
            "pellets": pellets,
            "pacman": pacman,
            "ghosts": ghosts,
            "power_pellets": power_pellets
        }
    