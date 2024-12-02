import argparse
import re
import random
import time  # Import the time module
from mcts_alphaZero import MCTSPlayer as MCTS_Alpha
from Fictitious_Agent import Fictitious_Agent, TRAINING_PARAMETERS, get_fpa_param_key
from game import Board, Game
from policy_value_net_pytorch import PolicyValueNet
from policy_value_net_pytorch_ResNet_bottleneck import PolicyValueNet as PolicyValueNetBottleneck  # Updated to bottleneck model
from enum import Enum
import pickle
import os
from policy_value_net_numpy import PolicyValueNetNumpy as TheanoPolicyValueNet

FPA_SIMULATIONS = 20


class BenchmarkModel(Enum):
    originalAgent = 1
    originalNN = 2

    def __str__(self):
        return self.name

def main(benchmark, board_size, mode):
    # Start timer
    start_time = time.time()
    
    # Extract board size and win condition from the board_size flag (e.g., "8_8_5")
    match = re.match(r"(\d+)_(\d+)_(\d+)", board_size)
    if match:
        width, height, n_in_row = map(int, match.groups())
    else:
        raise ValueError("Board size must be in the format 'WxH_N' (e.g., '8_8_5').")

    # Initialize the board and game
    board = Board(width=width, height=height, n_in_row=n_in_row)
    game = Game(board)

    # Determine model file and policy network based on benchmark and board size
    if benchmark == "originalAgent":
        model_file = f"best_policy_bottleneck_{board_size}.model"
        best_policy = PolicyValueNetBottleneck(width, height, model_file)
        benchmark_player = MCTS_Alpha(
            best_policy.policy_value_fn,
            c_puct=5,
            n_playout=100
        )
        benchmark_name = "originalAgent"
    elif benchmark == "originalNN":
        model_file = f"best_policy_{board_size}.model"
        try:
            with open(model_file, 'rb') as f:
                try:
                    # Attempt to load as a pickle file
                    os.environ['THEANO_FLAGS'] = 'device=cuda,floatX=float32'
                    policy_param = pickle.load(f, encoding='latin1')
                    print(f"Loaded model file '{model_file}' as a pickle.")
                    best_policy = TheanoPolicyValueNet(width, height, policy_param)
                except pickle.UnpicklingError:
                    # If pickle fails, try loading as a PyTorch model
                    f.seek(0)  # Reset file pointer to the beginning
                    policy_param = torch.load(f, map_location=torch.device('cuda'), weights_only=True)  # Load PyTorch model
                    print(f"Loaded model file '{model_file}' as a PyTorch model.")
                    best_policy = PytorchPolicyValueNet(width, height)
                    best_policy.policy_value_net.load_state_dict(policy_param)
        except FileNotFoundError:
            raise ValueError(f"Model file '{model_file}' not found.")
        benchmark_player = Fictitious_Agent(
            policy_value_function=best_policy.policy_value_fn,
            self_play=False,
            simulations=FPA_SIMULATIONS,
            depth=TRAINING_PARAMETERS[get_fpa_param_key(width, height, n_in_row)]['depth'],
            action_sample_count=TRAINING_PARAMETERS[get_fpa_param_key(width, height, n_in_row)]['action_sample_count']
        )
        benchmark_name = "originalNN"
    else:
        raise ValueError("Invalid benchmark. Use 'originalAgent' or 'originalNN'.")

    # Player 2: Opponent using Fictitious Agent and PolicyValueNetBottleneck
    model_file_2 = f"best_policy_bottleneck_{board_size}.model"
    best_policy_2 = PolicyValueNetBottleneck(width, height, model_file_2)
    opponent_player = Fictitious_Agent(
        policy_value_function=best_policy_2.policy_value_fn,
        self_play=False,
        simulations=FPA_SIMULATIONS,
        depth=TRAINING_PARAMETERS[get_fpa_param_key(width, height, n_in_row)]['depth'],
        action_sample_count=TRAINING_PARAMETERS[get_fpa_param_key(width, height, n_in_row)]['action_sample_count']
    )
    opponent_name = "combined"
    # Determine mode of operation: demonstration or rating
    if mode == "dem":
        print(f"Demonstrating a game between {benchmark} and the opponent.")
        winner = game.start_play(benchmark_player, opponent_player, start_player=0, is_shown=1)
        if winner == benchmark_player.player:
            print(f"Game over. Winner: {benchmark_name}")
        elif winner == opponent_player.player:
            print(f"Game over. Winner: {opponent_name}")
        else:
            print("Game over. Winner: Tie")
    elif mode == "rate":
        print(f"Rating {benchmark_name} on board size {board_size} against {opponent_name} over 20 games.")
        benchmark_win_count = 0
        combined_win_count = 0
        for _ in range(20):
            winner = game.start_play(benchmark_player, opponent_player, start_player=random.randint(0, 1), is_shown=0)
            if winner == benchmark_player.player:
                benchmark_win_count += 1
            elif winner == opponent_player.player:
                combined_win_count += 1
        benchmark_win_rate = benchmark_win_count / 20 * 100
        combined_win_rate = combined_win_count / 20 * 100
        print(f"Win rate of {benchmark_name} on board size {board_size}: {benchmark_win_rate}%")
        print(f"Win rate of {opponent_name} on board size {board_size}: {combined_win_rate}%")
    else:
        raise ValueError("Invalid mode. Use 'dem' for demonstration or 'rate' for rating.")
    
    # End timer and print elapsed time
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True, choices=["originalAgent", "originalNN"], help="Benchmark type: 'originalAgent' or 'originalNN'")
    parser.add_argument("--board_size", type=str, required=True, help="Board size in the format 'WxH_N' (e.g., '8_8_5')")
    parser.add_argument("--mode", type=str, required=True, choices=["dem", "rate"], help="Mode of operation: 'dem' or 'rate'")
    return parser

if __name__ == "__main__":
    # Parse arguments and execute the main function
    parser = get_args()
    args = parser.parse_args()
    main(args.benchmark, args.board_size, args.mode)
