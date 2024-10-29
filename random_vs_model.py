# random_vs_model.py
import re
import random
import numpy as np
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy
import pickle

class RandomPlayer:
    """A random move player, structured similarly to MCTSPlayer."""

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        pass  # No reset needed for random moves

    def get_action(self, board, temp=1e-3, return_prob=0):
        """Select a random action from available moves."""
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = random.choice(sensible_moves)
            move_probs = np.zeros(board.width * board.height)
            move_probs[move] = 1  # 100% probability on the chosen random move
            return (move, move_probs) if return_prob else move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "Random {}".format(self.player)


def main(model_name, mode):
    # Extract board size and win condition from the model name (e.g., "best_policy_8_8_5")
    match = re.search(r"_(\d+)_(\d+)_(\d+)", model_name)
    if match:
        width, height, n_in_row = map(int, match.groups())
    else:
        raise ValueError("Model name must contain board dimensions and win condition in the format '_WxH_N'.")

    # Initialize the board and game
    board = Board(width=width, height=height, n_in_row=n_in_row)
    game = Game(board)

    # Load the model
    model_file = f"{model_name}.model"
    try:
        with open(model_file, 'rb') as f:
            policy_param = pickle.load(f, encoding='latin1')


    except FileNotFoundError:
        raise ValueError(f"Model file '{model_file}' not found.")
    
    best_policy = PolicyValueNetNumpy(width, height, policy_param)
    mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

    # Initialize the random player
    random_player = RandomPlayer()

    # Determine mode of operation: demonstration or rating
    if mode == "dem":
        print(f"Demonstrating a game between {model_name} and RandomPlayer.")
        winner = game.start_play(random_player, mcts_player, start_player=0, is_shown=1)
        print("Game over. Winner:", "Random" if winner == random_player.player else "Model" if winner == mcts_player.player else "Tie")

    elif mode == "rate":
        print(f"Rating {model_name} against RandomPlayer over 100 games.")
        win_count = 0
        for _ in range(100):
            winner = game.start_play(random_player, mcts_player, start_player=random.randint(0, 1), is_shown=0)
            if winner == mcts_player.player:
                win_count += 1
        win_rate = win_count / 100 * 100
        print(f"Win rate of {model_name} against RandomPlayer: {win_rate}%")

    else:
        raise ValueError("Invalid mode. Use 'dem' for demonstration or 'rate' for rating.")

if __name__ == "__main__":
    # Example usage:
    # main("best_policy_8_8_5", "dem") for demonstration
    # main("best_policy_8_8_5", "rate") for rating
    import sys
    if len(sys.argv) != 3:
        print("Usage: python random_vs_model.py <model_name> <mode>")
        print("Example: python random_vs_model.py best_policy_8_8_5 dem")
    else:
        main(sys.argv[1], sys.argv[2])
