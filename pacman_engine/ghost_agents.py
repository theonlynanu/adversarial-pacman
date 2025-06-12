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
    def __init__(self, index):
        super().__init__(index)

    def get_action(self, state):
        start = state.get_ghost_position(self.index)
        goal = state.get_pacman_position()

        path = self.a_star_search(state, start, goal)

        if len(path) >= 2:
            next_pos = path[1]
            actions = Actions.get_legal_actions(state.get_ghost_state(self.index).configuration, state.get_walls())
            for action in actions:
                vector = Actions.direction_to_vector(action)
                successor = (int(start[0] + vector[0]), int(start[1] + vector[1]))
                if successor == next_pos:
                    return action

        # Fallback if path is empty
        return Directions.STOP

    def a_star_search(self, state, start, goal):
        frontier = PriorityQueue()
        frontier.push((start, []), 0)
        visited = set()

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
                    frontier.push((neighbor, new_path), cost)

        return [start]  # fallback if no path found
