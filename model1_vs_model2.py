# model1_vs_model2.py
import re
import random
import numpy as np
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy as TheanoPolicyValueNet
# from policy_value_net_pytorch_ResNet import PolicyValueNet as PytorchPolicyValueNet
from policy_value_net_pytorch_ResNet_bottleneck import PolicyValueNet as PytorchPolicyValueNet
import pickle
import torch
import os
from tqdm import tqdm
import sys
import time  # Import the time module


def extract_dimensions(model_name):
    """Extract board dimensions and win condition from the model name."""
    match = re.search(r"_(\d+)_(\d+)_(\d+)", model_name)
    if match:
        return map(int, match.groups())
    else:
        raise ValueError(f"Model name '{model_name}' must contain board dimensions and win condition in the format '_WxH_N'.")


def load_model(model_name, width, height):
    """Load model based on the file type."""
    model_file = f"{model_name}.model"
    try:
        with open(model_file, 'rb') as f:
            try:
                # Attempt to load as a pickle file
                os.environ['THEANO_FLAGS'] = 'device=cuda,floatX=float32'
                policy_param = pickle.load(f, encoding='latin1')
                print(f"Loaded model file '{model_name}' as a pickle.")
                return TheanoPolicyValueNet(width, height, policy_param)
            except pickle.UnpicklingError:
                # If pickle fails, try loading as a PyTorch model
                f.seek(0)  # Reset file pointer to the beginning
                policy_param = torch.load(f, map_location=torch.device('cuda'), weights_only=True)  # Load PyTorch model
                print(f"Loaded model file '{model_name}' as a PyTorch model.")
                best_policy = PytorchPolicyValueNet(width, height)
                best_policy.policy_value_net.load_state_dict(policy_param)
                return best_policy
    except FileNotFoundError:
        raise ValueError(f"Model file '{model_file}' not found.")


def main(model_name1, model_name2, mode):
    # Extract board dimensions and win condition from model names
    print(f"Loading model: {model_name1}")
    width1, height1, n_in_row1 = extract_dimensions(model_name1)
    print(f"Loading model: {model_name2}")
    width2, height2, n_in_row2 = extract_dimensions(model_name2)

    # Validate dimensions
    if (width1, height1, n_in_row1) != (width2, height2, n_in_row2):
        raise ValueError(f"Model dimensions do not match: {model_name1} ({width1}x{height1}, {n_in_row1}) vs {model_name2} ({width2}x{height2}, {n_in_row2}).")

    # Initialize the board and game
    board = Board(width=width1, height=height1, n_in_row=n_in_row1)
    game = Game(board)

    # Load models
    model1 = load_model(model_name1, width1, height1)
    model2 = load_model(model_name2, width2, height2)

    # Initialize MCTS players
    mcts_player1 = MCTSPlayer(model1.policy_value_fn, c_puct=5, n_playout=400)
    mcts_player2 = MCTSPlayer(model2.policy_value_fn, c_puct=5, n_playout=400)

    # Determine mode of operation: demonstration or rating
    if mode == "dem":
        print(f"Demonstrating a game between Player 1({model_name1}) and Player 2({model_name2}).")
        winner = game.start_play(mcts_player1, mcts_player2, start_player=0, is_shown=1)
        print("The model name for Winner is:", model_name1 if winner == mcts_player1.player else model_name2 if winner == mcts_player2.player else "Tie")

    elif mode == "rate":
        num_games = 20
        print(f"Rating {model_name1} against {model_name2} over {num_games} games.")
        win_count1 = 0
        win_count2 = 0
        for _ in tqdm(range(num_games), desc="Progress", unit="game"):
            winner = game.start_play(mcts_player1, mcts_player2, start_player=random.randint(0, 1), is_shown=0)
            if winner == mcts_player1.player:
                win_count1 += 1
            elif winner == mcts_player2.player:
                win_count2 += 1
        win_rate1 = win_count1 / num_games * 100
        win_rate2 = win_count2 / num_games * 100
        print(f"Win rate of {model_name1}: {win_rate1}%")
        print(f"Win rate of {model_name2}: {win_rate2}%")

    else:
        raise ValueError("Invalid mode. Use 'dem' for demonstration or 'rate' for rating.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python model1_vs_model2.py <model_name1> <model_name2> <mode>")
        print("Example: python model1_vs_model2.py best_policy_8_8_5 best_policy_8_8_5_ResNet dem")
    else:
        start_time = time.time()
        main(sys.argv[1], sys.argv[2], sys.argv[3])
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Total running time: {elapsed_time:.2f} seconds")
