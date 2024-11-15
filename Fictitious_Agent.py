"""
Fictitious Play Agent
"""
from typing import List, Dict

import numpy as np
#
# set_player_ind
# reset_player
# get_action
# move_to_location

class State_Node(object):
    """A node in the game tree"""
    def __init__(self, action_priors: Dict, available_actions: List):
        # maps from action to child nodes
        self.children = {}
        # maps from action to action counts, initialized to 1
        self.action_counts = {action: 1 for action in available_actions}
        self.total_actions = sum(self.action_counts.values())
        self.action_probabilities = self.get_action_probabilities()
        # maps from action to prior policy probability (from current policy network)
        self.action_priors = action_priors

    def get_action_probabilities(self):
        """Initialize action probabilities for a state action"""
        return {action: action_count/self.total_actions for action, action_count in self.action_counts.items()}

    def update_actions(self, select_action):
        """
        Update the action counts and action probabilities after an action selection
        """
        self.total_actions += 1
        self.action_counts[select_action] += 1
        for action, action_probability in self.action_probabilities.items():
            self.action_probabilities[action] = self.action_counts[action]/self.total_actions

class Ficticious_Agent(object):
