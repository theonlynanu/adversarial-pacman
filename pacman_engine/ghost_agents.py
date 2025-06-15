# ghost_agents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman_engine.game import Agent
from pacman_engine.game import Actions
from pacman_engine.game import Directions
import random
from pacman_engine.util import manhattan_distance, PriorityQueue
import pacman_engine.util as util



class GhostAgent(Agent):
    def __init__(self, index):
        super().__init__(index)

    def get_action(self, state):
        dist = self.get_distribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.choose_from_distribution(dist)

    def get_distribution(self, state):
        """Returns a Counter encoding a distribution over actions from the provided state."""
        util.raise_not_defined()


class RandomGhost(GhostAgent):
    """A ghost that chooses a legal action uniformly at random."""

    def get_distribution(self, state):
        dist = util.Counter()
        for a in state.get_legal_actions(self.index):
            dist[a] = 1.0
        dist.normalize()
        return dist


class DirectionalGhost(GhostAgent):
    """A ghost that prefers to rush Pacman, or flee when scared."""

    def __init__(self, index, prob_attack=0.8, prob_scared_flee=0.8):
        super().__init__(index)
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scared_flee

    def get_distribution(self, state):
        # Read variables from state
        ghost_state = state.getGhostState(self.index)
        legal_actions = state.get_legal_actions(self.index)
        pos = state.getGhostPosition(self.index)
        is_scared = ghost_state.scaredTimer > 0

        speed = 1
        if is_scared:
            speed = 0.5

        action_vectors = [Actions.directionToVector(a, speed) for a in legal_actions]
        new_positions = [(pos[0]+a[0], pos[1]+a[1]) for a in action_vectors]
        pacman_position = state.getPacmanPosition()

        # Select best actions given the state
        distances_to_pacman = [manhattan_distance(pos, pacman_position) for pos in new_positions]
        if is_scared:
            best_score = max(distances_to_pacman)
            best_prob = self.prob_scaredFlee
        else:
            best_score = min(distances_to_pacman)
            best_prob = self.prob_attack
        best_actions = [action for action, distance in zip(legal_actions, distances_to_pacman)
                        if distance == best_score]

        # Construct distribution
        dist = util.Counter()
        for a in best_actions:
            dist[a] = best_prob / len(best_actions)
        for a in legal_actions:
            dist[a] += (1-best_prob) / len(legal_actions)
        dist.normalize()
        return dist


class AStarGhost(Agent):
    def __init__(self, index, shared_info=None):
        super().__init__(index)
        self.shared_info = shared_info if shared_info is not None else {}

    def get_action(self, state):
        start = state.get_ghost_position(self.index)
        ghost_state = state.get_ghost_state(self.index)

        if ghost_state.scared_timer > 0:
            # Ghost is scared — run away from Pacman
            goal = self._farthest_legal_tile(state, start, state.get_pacman_position())
        else:
            goal = state.get_pacman_position()


        # Read other ghost’s path
        other_index = 2 if self.index == 1 else 1
        other_path = self.shared_info.get(f"path_{other_index}", [])

        # Plan path with A*
        path = self.a_star_search(state, start, goal)

        # If both ghosts try to go to the same next spot, offset this one
        if len(path) >= 2 and len(other_path) >= 2 and path[1] == other_path[1]:
            # Slight detour to avoid same spot
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

        return Directions.STOP

    def _farthest_legal_tile(self, state, start, pacman_pos):
        walls = state.get_walls()
        neighbors = Actions.get_legal_neighbors(start, walls)
        farthest = start
        max_dist = -1
        for n in neighbors:
            dist = manhattan_distance(n, pacman_pos)
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

