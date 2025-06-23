from pacman_engine.game import Agent
from pacman_engine.game import Actions
from pacman_engine.game import Directions
from pacman_engine.pacman import GameState
import random
import json, gzip, pickle, os

from pacman_engine.util import manhattan_distance, PriorityQueue


DIRECTIONS_TO_IDX = {
    'North': 0,
    'South': 1,
    'East': 2,
    'West': 3,
}

ACTIONS = [
    Directions.NORTH,
    Directions.SOUTH,
    Directions.EAST,
    Directions.WEST,
    Directions.STOP
    ]

DISTANCE_BUCKETS = {
    # Shows the max manhattan distance for each bucket key
    0: 1,
    1: 2,
    2: 4,
    3: float('inf')
}

TIMER_BUCKETS = {
    0: 0,
    1: 5,
    2: 15,
    3: float('inf')
}

NUM_ACTIONS_PACMAN = 5  # N,S,E,W,Stop

OPTIMISTIC_Q = 0.0

def bucket_distance(d):
    """ Returns the bucket 0-3 for a given Manhattan Distance"""
    for bucket, distance in DISTANCE_BUCKETS.items():
        if d <= distance:
            return bucket
        
    return max(DISTANCE_BUCKETS.keys())
        
def bucket_timer(t):
    """ Returns the bucket 0-3 for a given timer value"""
    for bucket, time in TIMER_BUCKETS.items():
        if t <= time:
            return bucket
        
    return max(TIMER_BUCKETS.keys())
        

class AStarGhost(Agent):
    def __init__(self, index, shared_info=None):
        super().__init__(index)
        self.shared_info = shared_info if shared_info is not None else {}

    def get_action(self, state):
        # RE-READ the updated ghost state (ensure this comes *after* Pac-Man move)
        ghost_state = state.get_ghost_state(self.index)
        start = ghost_state.get_position()
        pacman_pos = state.get_pacman_position()
        
        other_index = 2 if self.index == 1 else 1
        other_path = self.shared_info.get(f"path_{other_index}", [])

        # Re-check scared status
        if ghost_state.scared_timer > 0:
            goal = self._farthest_legal_tile(state, start, pacman_pos)
            path = self.a_star_search(state, start, goal)
        else:
            goal = pacman_pos
            path = self.a_star_search(state, start, goal)

            if len(path) >= 2 and len(other_path) >= 2 and path[1] == other_path[1]:
                neighbors = Actions.get_legal_neighbors(start, state.get_walls())
                for n in neighbors:
                    if n != other_path[1]:
                        path = [start, n]
                        break
                    
        # Save this ghost’s path
        self.shared_info[f"path_{self.index}"] = path

        # Convert next step to action
        if len(path) >= 2:
            next_pos = path[1]
            actions = state.get_legal_actions(self.index)
            for action in actions:
                vector = Actions.direction_to_vector(action)
                successor = (int(start[0] + vector[0]), int(start[1] + vector[1]))
                if successor == next_pos:
                    return action

        legal_actions = state.get_legal_actions(self.index)
        if legal_actions:
            return random.choice(legal_actions)
        else:
            return Directions.STOP 

    def _farthest_legal_tile(self, state, start, pacman_pos):
        walls = state.get_walls()
        neighbors = Actions.get_legal_neighbors(start, walls)
        farthest = start
        max_dist = -1
        for n in neighbors:
            dist = manhattan_distance(n, pacman_pos)
            # 🚫 Reject positions that move ghost *closer* to Pacman
            if dist > manhattan_distance(start, pacman_pos):
                if dist > max_dist:
                    farthest = n
                    max_dist = dist
        return farthest


    def a_star_search(self, state, start, goal):
        frontier = PriorityQueue()
        frontier.push((start, []), 0)
        visited = set()
        
        # Get the path of the other ghost to avoid overlapping
        other_index = 2 if self.index == 1 else 1
        other_path = self.shared_info.get(f"path_{other_index}", [])


        while not frontier.is_empty():
            current_pos, path = frontier.pop()

            if current_pos in visited:
                continue
            visited.add(current_pos)

            if current_pos == goal:
                return path + [current_pos]

            for neighbor in Actions.get_legal_neighbors(current_pos, state.get_walls()):
                if neighbor not in visited:
                    new_path = path + [current_pos]
                    cost = len(new_path) + manhattan_distance(neighbor, goal)

                    # Penalize overlap with other ghost's path
                    if neighbor in other_path:
                        cost += 3  # You can tune this value for better separation

                    frontier.push((neighbor, new_path), cost)

        return [start]  # fallback if no path found


class QPacman(Agent):
    def __init__(self,
                 gamma = 0.99,
                 epsilon = 1,
                 epsilon_min = 0.0,
                 decay_rate = 0.9999,
                 ):
        super().__init__(index = 0)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        
        
        self.Q: dict[tuple[tuple, str]: float] = {}                 # (state_hash, action): Q Value  
        self.visited_sa: dict[tuple[tuple, str]: int] = {}          # (state_hash, action): observed_count                      
        
    """ ========================= Q LEARNING METHODS ========================= """    
        
        
    def get_action(self, state: GameState):
        """ Required by the Agent superclass - 
        
            Originally chose an action via epsilon greedy
            using an adaptive epsilon, e_0 / (1 + state_visits), but decided
            to go with a traditional decay rate
        """
        s = self._state_key(state)
        legal = state.get_legal_actions(self.index)
        
        # Removes stop unless it's the only move left
        if Directions.STOP in legal and len(legal) > 1:
            legal.remove(Directions.STOP)
        
        if not legal:
            print("No legal moves?")
            return Directions.STOP          # Shouldn't really occur, but gives us a way to see if there's a error in logic
        
        # Evaluate epsilon-greedy strategy
        take_random = random.random() < self.epsilon
        
        if take_random:
            return random.choice(legal)
        
        # q_best, a_best = float('-inf'), None
        
        # for a in legal:
        #     q_val = self.Q.get((s, a), 0.0)
        #     if q_val > q_best:
        #         q_best, a_best = q_val, a

        # return a_best
        
        # Take Greedy approach
        q_vals = [(a, self.Q.get((s, a), OPTIMISTIC_Q)) for a in legal]
        max_q = max(v for _, v in q_vals)
        
        
        best_actions = [a for a, v in q_vals if v == max_q]
        return random.choice(best_actions)
    
    
    def observe_transition(self,
                           state: GameState,
                           action,
                           next_state: GameState,
                           reward):
        """Called after the environment applies the action to get to a new state and
        generate the reward. """
        
        
        s_key = self._state_key(state)
        s_key_n = self._state_key(next_state)
        
        if (s_key, action) not in self.Q:
            self.Q[(s_key, action)] = OPTIMISTIC_Q
        
        self.visited_sa[(s_key, action)] = self.visited_sa.get((s_key, action), 0) + 1
        
        # eta = 1.0 / self.visited_sa[(s_key, action)]
        
        # TRYING A FIXED ALPHA
        eta = 0.2
        
        
        # visits_sa = self.visited_sa[(s_key, action)]
        # if visits_sa < 200:
        #     eta = 0.5
        # elif visits_sa < 1000:
        #     eta = 0.1
        # else:
        #     eta = 0.05
        
        max_q_n = max(
            (self.Q.get((s_key_n, a_prime), OPTIMISTIC_Q)
             for a_prime in next_state.get_legal_actions(self.index)),
            default = OPTIMISTIC_Q
        )
        
        target = reward + self.gamma * max_q_n
        old_q = self.Q[(s_key, action)]
        new_q = old_q + eta * (target - old_q)
        self.Q[(s_key, action)] = new_q
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.decay_rate, self.epsilon_min)
        
        
    """ ----------------------------- HELPER METHODS ----------------------------- """        
        
    def _state_key(self, state: GameState):
        """Takes a GameState and returns a tuple based on the feature set:
        - Nearest ghost direction - 4 values [0,3] - general directions N, S, E, W. I want to start with just cardinals and then expand to NE, SE, NW, SW if we find that isn’t enough information
        - Nearest ghost distance - 4 values [0,3] - bucketed distance
        - Scared timer of nearest ghost - 4 values [0, 3] - bucketed scared timer for just the nearest ghost
        - Nearest pellet direction - 4 values [0, 3] - cardinal direction of nearest pellet
        - Nearest pellet distance - 4 values [0,3] - bucketed distance to nearest pellet
        - Power pellets present - 2 values 0 or 1 - flag for if there are power pellets left on the board
        - Nearest power pellet direction - 4 values [0,3] - cardinal direction to nearest power pellet, only if the flag is 1
        - Nearest power pellet distance - 4 values [0,3] - bucketed distance to nearest power pellet, only if the flag is 1
        - Surrounding walls - 16 values in a 4-bit mask - immediate surrounding walls to determine legal moves
        """
        
        pac = state.get_pacman_position()
        walls_mask = self._walls_mask(state)
        
        ghost_dir_b, ghost_dist_b, ghost_timer_b = self._nearest_ghost_features(
            pac, state.get_ghost_states()
        )
        
        pellet_dir_b, pellet_dist_b = self._nearest_food_features(pac, state.get_food())
        
        power_pellets = state.get_capsules()
        power_present = 1 if power_pellets else 0
        
        if power_present:
            p_dir_b, p_dist_b = self._nearest_capsule_features(pac, power_pellets)
            return (
                ghost_dir_b, ghost_dist_b, ghost_timer_b,
                pellet_dir_b, pellet_dist_b,
                power_present,
                p_dir_b, p_dist_b,
                walls_mask
            )
            
        else:
            return (
                ghost_dir_b, ghost_dist_b, ghost_timer_b,
                pellet_dir_b, pellet_dist_b,
                power_present,
                walls_mask
            )


    def _get_cardinal(self, src, dest):
        """
        Maps relative positions to a direction from 0-3
        0 - North
        1 - South
        2 - East
        3 - West
        """
        dx = dest[0] - src[0]
        dy = dest[1] - src[1]
        
        if abs(dy) >= abs(dx):
            return 0 if dy > 0 else 1
        else:
            return 2 if dx > 0 else 3
        
        
    def _walls_mask(self, gamestate: GameState):
        """Generates a 4-bit mask of walls around pacman
        N, S, E, W
        """
        
        px, py = map(int, gamestate.get_pacman_position())
        walls = gamestate.get_walls()
        h, w = walls.height, walls.width
        
        inside = lambda x,y: 0 <= x < w and 0 <= y < h
        mask = 0
        
        
        if inside(px, py+1) and walls[px][py+1]:mask |= walls[px][py + 1] << 3  # Wall to north side (bit 3)
        if inside(px, py-1) and walls[px][py-1]:mask |= walls[px][py - 1] << 2  # Wall to south side (bit 2)
        if inside(px+1, py) and walls[px+1][py]:mask |= walls[px + 1][py] << 1  # Wall to East side (bit 1)
        if inside(px-1, py) and walls[px-1][py]:mask |= walls[px - 1][py]       # Wall to West side (bit 0)
        
        return mask                     # 0-15
        
                
    def _nearest_ghost_features(self, pac_pos, ghost_states):
        """Returns (direction bucket, distance bucket, timer bucket) for nearest ghost"""
    
        if not ghost_states:
            return 0, 3, 0      # Default feature set saying it is very far away
        
        
        best_dist = float("inf")
        best_state = None
        for ghost in ghost_states:
            dist = manhattan_distance(pac_pos, ghost.get_position())
            if dist < best_dist:
                best_dist, best_state = dist, ghost
                
        dir_buck = self._get_cardinal(pac_pos, best_state.get_position())
        dist_buck = bucket_distance(best_dist)
        timer_buck = bucket_timer(best_state.scared_timer)
        return dir_buck, dist_buck, timer_buck
    
    
    def _nearest_food_features(self, pac_pos, food_grid):
        """ Returns nearest pellet features according to the food_grid
        food_grid should be a Grid object"""
        
        nearest, best_dist = None, float("inf")
        
        for x in range(food_grid.width):
            for y in range(food_grid.height):
                if food_grid[x][y]:
                    dist = abs(x - pac_pos[0]) + abs(y - pac_pos[1])
                    if dist < best_dist:
                        best_dist, nearest = dist, (x, y)
                        
        if nearest is None:         # Shouldn't really occur since game ends when all pellets are gone
            return 0, 3         
        
        dir_buck = self._get_cardinal(pac_pos, nearest)
        dist_buck = bucket_distance(best_dist)
        return dir_buck, dist_buck
    
    
    def _nearest_capsule_features(self, pac_pos, capsules):
        """ Capsules is expected to be a list of (x, y) positions
            Returns direction and distance buckets to the nearest pellet
        """
        
        nearest, best_dist = None, float('inf')
        
        for (x, y) in capsules:
            dist = abs(x - pac_pos[0]) + abs(y - pac_pos[1])
            if dist < best_dist:
                best_dist, nearest = dist, (x, y)
                
        dir_buck = self._get_cardinal(pac_pos, nearest)
        dist_buck = bucket_distance(best_dist)
        
        return dir_buck, dist_buck

    """ ~~~~~~~~~~~~~~~ UTILITY METHODS ~~~~~~~~~~~~~~~ """
    def save(self, path: str):
        """ Stores Q table and visited_sa. Uses gzip-pickle to compress large
        Q table"""
        
        with gzip.open(path, "wb") as file:
            pickle.dump(
                {
                "Q": self.Q,
                "visited_sa": self.visited_sa
                },
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            
    @classmethod    
    def load(cls, path: str, **kw):
        """ Factory method for creating a new QPacman Instance with loaded tables"""
        agent = cls(**kw)
        if os.path.isfile(path):
            with gzip.open(path, 'rb') as file:
                blob = pickle.load(file)
            agent.Q = blob["Q"]
            agent.visited_sa = blob["visited_sa"]
            
        return agent


REL_ACTIONS = ["Forward", "Left", "Right", "Reverse", "Stop"]

REL_TO_ABS = {
    Directions.NORTH: {
        "Forward": Directions.NORTH,
        "Left"   : Directions.WEST,
        "Right"  : Directions.EAST,
        "Reverse": Directions.SOUTH,
    },
    Directions.SOUTH: {
        "Forward": Directions.SOUTH,
        "Left"   : Directions.EAST,
        "Right"  : Directions.WEST,
        "Reverse": Directions.NORTH,
    },
    Directions.EAST : {
        "Forward": Directions.EAST,
        "Left"   : Directions.NORTH,
        "Right"  : Directions.SOUTH,
        "Reverse": Directions.WEST,
    },
    Directions.WEST : {
        "Forward": Directions.WEST,
        "Left"   : Directions.SOUTH,
        "Right"  : Directions.NORTH,
        "Reverse": Directions.EAST,
    },
    Directions.STOP : {  # when starting a game
        "Forward": Directions.NORTH,
        "Left"   : Directions.WEST,
        "Right"  : Directions.EAST,
        "Reverse": Directions.SOUTH,
    },
}

def rel_to_abs(curr_dir: str, rel_move: str):
    return REL_TO_ABS[curr_dir][rel_move] if rel_move != "Stop" else Directions.STOP

class QPacmanRelative(Agent):
    """ The same as QPacman but using relative directions rather than absolute 
    directions as actions"""
    def __init__(self,
                 gamma = 0.99,
                 epsilon = 1,
                 epsilon_min = 0.0,
                 decay_rate = 0.9999,
                 ):
        super().__init__(index = 0)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        
        
        self.Q: dict[tuple[tuple, str]: float] = {}                 # (state_hash, action): Q Value  
        self.visited_sa: dict[tuple[tuple, str]: int] = {}          # (state_hash, action): observed_count                      
        
    """ ========================= Q LEARNING METHODS ========================= """    
        
        
    def get_action(self, state: GameState):
        """ Required by the Agent superclass - 
        
            Originally chose an action via epsilon greedy
            using an adaptive epsilon, e_0 / (1 + state_visits), but decided
            to go with a traditional decay rate
        """
        s = self._state_key(state)
        
        curr_heading = state.get_pacman_state().get_direction()
        abs_legals = state.get_legal_actions(self.index)
        rel_legals = []
        
        for rel_move in REL_ACTIONS:
            abs_move = rel_to_abs(curr_heading, rel_move)
            if abs_move in abs_legals:
                rel_legals.append(rel_move)
                
        if not rel_legals:
            return Directions.STOP      # Shouldn't occur but let's us observe errors
        
        if "Stop" in rel_legals and len(rel_legals) > 1:
            rel_legals.remove("Stop")
        
        # Evaluate epsilon-greedy strategy
        take_random = random.random() < self.epsilon
        
        if take_random:
            return rel_to_abs(curr_heading, random.choice(rel_legals))
        else:
            q_best = float("-inf")
            best_set = []
            for action in rel_legals:
                q = self.Q.get((s, action), OPTIMISTIC_Q)
                if q > q_best:
                    q_best, best_set = q, [action]
                elif q == q_best:
                    best_set.append(action)

            chosen_rel = random.choice(best_set)
            
            return rel_to_abs(curr_heading, chosen_rel)
    
    def observe_transition(self,
                           state: GameState,
                           action,
                           next_state: GameState,
                           reward):
        """Called after the environment applies the action to get to a new state and
        generate the reward. action is absolute while rel_action will be relative"""
        
        curr_heading = state.get_pacman_state().get_direction()
        rel_action = "Stop" if action == Directions.STOP else \
            next(k for k, v in REL_TO_ABS[curr_heading].items() if v == action)
            
        s_key = self._state_key(state)
        s_key_n = self._state_key(next_state)
        
        self.visited_sa[(s_key, rel_action)] = self.visited_sa.get((s_key, rel_action), 0) + 1
        
        if (s_key, rel_action) not in self.Q:
            self.Q[(s_key, rel_action)] = OPTIMISTIC_Q
        
        eta = 1.0 / self.visited_sa[(s_key, rel_action)]
        
        curr_heading_n = next_state.get_pacman_state().get_direction()
        abs_legals_n = next_state.get_legal_actions(self.index)
        rel_legals_n = [a for a in REL_ACTIONS if rel_to_abs(curr_heading_n, a) in abs_legals_n]
        
        max_q_n = max(
            (self.Q.get((s_key_n, a), OPTIMISTIC_Q) for a in rel_legals_n),
            default=OPTIMISTIC_Q
        )
        
        old_q = self.Q.get((s_key, rel_action), OPTIMISTIC_Q)
        self.Q[(s_key, rel_action)] = old_q + eta * (reward + self.gamma * max_q_n - old_q)
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.decay_rate, self.epsilon_min)
        
        
    """ ----------------------------- HELPER METHODS ----------------------------- """        
        
    def _state_key(self, state: GameState):
        """Takes a GameState and returns a tuple based on the feature set:
        - Nearest ghost direction - 4 values [0,3] - general directions N, S, E, W. I want to start with just cardinals and then expand to NE, SE, NW, SW if we find that isn’t enough information
        - Nearest ghost distance - 4 values [0,3] - bucketed distance
        - Scared timer of nearest ghost - 4 values [0, 3] - bucketed scared timer for just the nearest ghost
        - Nearest pellet direction - 4 values [0, 3] - cardinal direction of nearest pellet
        - Nearest pellet distance - 4 values [0,3] - bucketed distance to nearest pellet
        - Power pellets present - 2 values 0 or 1 - flag for if there are power pellets left on the board
        - Nearest power pellet direction - 4 values [0,3] - cardinal direction to nearest power pellet, only if the flag is 1
        - Nearest power pellet distance - 4 values [0,3] - bucketed distance to nearest power pellet, only if the flag is 1
        - Surrounding walls - 16 values in a 4-bit mask - immediate surrounding walls to determine legal moves
        """
        
        pac = state.get_pacman_position()
        heading = state.get_pacman_state().get_direction()
        
        walls_mask = self._walls_mask(state)
        
        ghost_dir_b, ghost_dist_b, ghost_timer_b = self._nearest_ghost_features(
            pac, state.get_ghost_states()
        )
        
        pellet_dir_b, pellet_dist_b = self._nearest_food_features(pac, state.get_food())
        
        power_pellets = state.get_capsules()
        power_present = 1 if power_pellets else 0
        
        if power_present:
            p_dir_b, p_dist_b = self._nearest_capsule_features(pac, power_pellets)
            return (
                heading,
                ghost_dir_b, ghost_dist_b, ghost_timer_b,
                pellet_dir_b, pellet_dist_b,
                power_present,
                p_dir_b, p_dist_b,
                walls_mask
            )
            
        else:
            return (
                heading,
                ghost_dir_b, ghost_dist_b, ghost_timer_b,
                pellet_dir_b, pellet_dist_b,
                power_present,
                walls_mask
            )


    def _get_cardinal(self, src, dest):
        """
        Maps relative positions to a direction from 0-3
        0 - North
        1 - South
        2 - East
        3 - West
        """
        dx = dest[0] - src[0]
        dy = dest[1] - src[1]
        
        if abs(dy) >= abs(dx):
            return 0 if dy > 0 else 1
        else:
            return 2 if dx > 0 else 3
        
        
    def _walls_mask(self, gamestate: GameState):
        """Generates a 4-bit mask of walls around pacman
        N, S, E, W
        """
        
        px, py = map(int, gamestate.get_pacman_position())
        walls = gamestate.get_walls()
        h, w = walls.height, walls.width
        
        inside = lambda x,y: 0 <= x < w and 0 <= y < h
        mask = 0
        
        
        if inside(px, py+1) and walls[px][py+1]:mask |= walls[px][py + 1] << 3  # Wall to north side (bit 3)
        if inside(px, py-1) and walls[px][py-1]:mask |= walls[px][py - 1] << 2  # Wall to south side (bit 2)
        if inside(px+1, py) and walls[px+1][py]:mask |= walls[px + 1][py] << 1  # Wall to East side (bit 1)
        if inside(px-1, py) and walls[px-1][py]:mask |= walls[px - 1][py]       # Wall to West side (bit 0)
        
        return mask                     # 0-15
        
                
    def _nearest_ghost_features(self, pac_pos, ghost_states):
        """Returns (direction bucket, distance bucket, timer bucket) for nearest ghost"""
    
        if not ghost_states:
            return 0, 3, 0      # Default feature set saying it is very far away
        
        
        best_dist = float("inf")
        best_state = None
        for ghost in ghost_states:
            dist = manhattan_distance(pac_pos, ghost.get_position())
            if dist < best_dist:
                best_dist, best_state = dist, ghost
                
        dir_buck = self._get_cardinal(pac_pos, best_state.get_position())
        dist_buck = bucket_distance(best_dist)
        timer_buck = bucket_timer(best_state.scared_timer)
        return dir_buck, dist_buck, timer_buck
    
    
    def _nearest_food_features(self, pac_pos, food_grid):
        """ Returns nearest pellet features according to the food_grid
        food_grid should be a Grid object"""
        
        nearest, best_dist = None, float("inf")
        
        for x in range(food_grid.width):
            for y in range(food_grid.height):
                if food_grid[x][y]:
                    dist = abs(x - pac_pos[0]) + abs(y - pac_pos[1])
                    if dist < best_dist:
                        best_dist, nearest = dist, (x, y)
                        
        if nearest is None:         # Shouldn't really occur since game ends when all pellets are gone
            return 0, 3         
        
        dir_buck = self._get_cardinal(pac_pos, nearest)
        dist_buck = bucket_distance(best_dist)
        return dir_buck, dist_buck
    
    
    def _nearest_capsule_features(self, pac_pos, capsules):
        """ Capsules is expected to be a list of (x, y) positions
            Returns direction and distance buckets to the nearest pellet
        """
        
        nearest, best_dist = None, float('inf')
        
        for (x, y) in capsules:
            dist = abs(x - pac_pos[0]) + abs(y - pac_pos[1])
            if dist < best_dist:
                best_dist, nearest = dist, (x, y)
                
        dir_buck = self._get_cardinal(pac_pos, nearest)
        dist_buck = bucket_distance(best_dist)
        
        return dir_buck, dist_buck

    """ ~~~~~~~~~~~~~~~ UTILITY METHODS ~~~~~~~~~~~~~~~ """
    def save(self, path: str):
        """ Stores Q table and visited_sa. Uses gzip-pickle to compress large
        Q table"""
        
        with gzip.open(path, "wb") as file:
            pickle.dump(
                {
                "Q": self.Q,
                "visited_sa": self.visited_sa
                },
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            
    @classmethod    
    def load(cls, path: str, **kw):
        """ Factory method for creating a new QPacman Instance with loaded tables"""
        agent = cls(**kw)
        if os.path.isfile(path):
            with gzip.open(path, 'rb') as file:
                blob = pickle.load(file)
            agent.Q = blob["Q"]
            agent.visited_sa = blob["visited_sa"]
            
        return agent
