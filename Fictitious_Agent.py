"""
Fictitious Play Agent
"""
import copy
from random import random
from typing import List, Dict

import numpy as np

from game import Board

def get_fpa_param_key(width: int, height: int, n_in_row: int):
    return f"{width}_{height}_{n_in_row}"

TRAINING_PARAMETERS = {'6_6_4': {'n_playout': 20, # num of simulations for each move
                                 'game_batch_num': 500, # number of self-play games
                                 'batch_size': 500, # mini-batch size for training
                                 'check_freq': 30, # how often to evaluate the model
                                 'save_freq': 20, # how often to save the model
                                 'pure_mcts_playout_num': 100,
                                 'pure_mcts_step': 100,
                                 'pure_mcts_max_num': 500,
                                 'depth': 3,
                                 'action_sample_count': 5},
                       '8_8_5': {'n_playout': 20,
                                 'game_batch_num': 900,
                                 'batch_size': 500,
                                 'check_freq': 30,
                                 'save_freq': 20,
                                 'pure_mcts_playout_num': 100,
                                 'pure_mcts_step': 100,
                                 'pure_mcts_max_num': 500,
                                 'depth': 4,
                                 'action_sample_count': 8}
                       }

TIE = -1

def convert_action_priors(action_priors: List) -> Dict:
    # convert action_priors from tuples to dict
    return {action: action_probability for (action, action_probability) in action_priors}

def normalize_probabilities(probabilities: List):
    probs = np.array(probabilities)
    probs = probs/probs.sum()
    return probs

class StateNode(object):
    """A node in the game tree"""
    def __init__(self, board: Board, policy_value_function, action_sample_count):
        self.board = board
        self.policy_value_function = policy_value_function

        # get action prior from the policy network with just available moves
        action_priors, _ = self.policy_value_function(self.board)
        # maps from available action to its prior policy probability
        self.action_priors = convert_action_priors(action_priors)
        self.total_actions = len(self.action_priors)

        # maps from action to child nodes
        self.children = {}

        # initialize action_counts to 1 for all actions
        self.action_counts = {action: 1 for action in self.action_priors}
        self.action_sample_count = action_sample_count

    def sample_actions(self):
        return np.random.choice(list(self.action_priors.keys()),
                                p=normalize_probabilities(list(self.action_priors.values())),
                                size=self.action_sample_count,
                                replace=False)


    def get_action_probabilities(self):
        return {action: self.action_counts[action]/self.total_actions for action in self.action_priors}

    def get_expected_state_value(self, action_values: Dict):
        value = 0
        action_probabilities = self.get_action_probabilities()
        for action, probability in action_probabilities.items():
            value += probability * action_values.get(action, 0)
        return value


class Fictitious_Agent(object):
    def __init__(self, policy_value_function, self_play: bool, simulations: int, depth: int, action_sample_count:int):
        # policy_value_function takes a board object and outputs
        # 1) list of (action, probability) tuples
        # 2) state value in [-1, 1] in current player's perspective
        self.policy_value_function = policy_value_function
        # number of simulations to run for each action selection
        self.simulations = simulations
        self.depth = depth
        self.self_play = self_play
        self.root_node = None
        self.player = None
        self.action_sample_count = action_sample_count

    def __repr__(self):
        return "FictitiousAgent"

    @ staticmethod
    def get_winning_score(state_node: StateNode, winner: int):
        if winner == TIE:
            return 0
        elif winner == state_node.board.get_current_player():
            return 1
        else:
            return -1

    def action_search(self, state_node: StateNode, depth:int) -> float:
        # runs action selection search from a state node and returns expected state value

        end, winner = state_node.board.game_end()
        # if game ends, return the actual state values
        if end:
            return self.get_winning_score(state_node, winner)
        # return estimated value from value network if reaching the maximum depth
        elif depth == 0:
            _, state_value = self.policy_value_function(state_node.board)
            return state_value
        else:
            # randomly sample the actions to branch out from the prior probabilities to limit branch factor
            if len(state_node.action_priors) > self.action_sample_count:
                action_sample = state_node.sample_actions()
            else:
                # TODO: check this
                action_sample = state_node.board.availables

            action_state_values = dict()
            maximum_action_value = float('-inf')
            optimal_action = None

            for action in action_sample:
                save_last_move = state_node.board.last_move
                state_node.board.do_move(action)

                if action in state_node.children:
                    next_state = state_node.children[action]
                else:
                    next_state = StateNode(state_node.board, self.policy_value_function, self.action_sample_count)
                    state_node.children[action] = next_state

                next_state_value = self.action_search(next_state, depth - 1)
                # multiply by -1 since the return value is from the opponent's perspective
                next_state_value = -1 * next_state_value

                if next_state_value > maximum_action_value:
                    maximum_action_value = next_state_value
                    optimal_action = action

                state_node.board.remove_last_move(save_last_move)
                action_state_values[action] = next_state_value

            state_node.action_counts[optimal_action] += 1
            state_node.total_actions += 1
            state_value = state_node.get_expected_state_value(action_state_values)

            return state_value

    def run_simulations(self):
        state_node = self.root_node
        for simulation in range(self.simulations):
            state_value = self.action_search(state_node, self.depth)
        action_probs = state_node.get_action_probabilities()
        return action_probs

    def init_root_node(self, board):
        self.root_node = StateNode(board, self.policy_value_function, self.action_sample_count)

    def get_action(self, board, return_prob = False, temp=None):
        # TODO: delete temp if wanted
        # initialize root node
        if not self.root_node: self.init_root_node(board)

        action_probs = np.zeros(board.width * board.height)
        # maps actions to its policy prob
        policy = self.run_simulations()

        search_actions, search_probs = np.array(list(policy.keys())), np.array(list(policy.values()))
        # action probability data to use for training the networks
        action_probs[search_actions] = search_probs

        if self.self_play:
            # TODO: use dirichlet for now, change if needed
            select_action = np.random.choice(search_actions,
                                             p=0.75 * search_probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(search_probs)))
            )
            self.update_root_node(select_action)
        else:
            select_action = max(policy, key=policy.get)
            self.update_root_node(None)

        if return_prob:
            return select_action, action_probs
        else:
            return select_action

    def update_root_node(self, select_action):
        # reuse the subtree of root node during selfplay while resets the root node for regular playing
        if select_action is None or (select_action not in self.root_node.children):
            self.root_node = None
        else:
            self.root_node = self.root_node.children[select_action]


    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
       self.update_root_node(None)

