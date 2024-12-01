import argparse
import re
import random
from enum import Enum

from Fictitious_Agent import Fictitious_Agent, TRAINING_PARAMETERS, get_fpa_param_key
from game import Board, Game
from policy_value_net_pytorch_ResNet_bottleneck import PolicyValueNet
from random_vs_model import RandomPlayer
from mcts_pure import MCTSPlayer as MCTS_Pure  # Pure MCTS Player

class BenchmarkModel(Enum):
    Random = 1
    PureMcts = 2

    def __str__(self):
        return self.name

def get_benchmark_player(benchmark):
    """Return the appropriate benchmark player."""
    if benchmark == BenchmarkModel.Random:
        return RandomPlayer()
    elif benchmark == BenchmarkModel.PureMcts:
        return MCTS_Pure(c_puct=5, n_playout=100)
    else:
        raise ValueError("Invalid benchmark model. Choose from Random or PureMcts.")

def main(model_name, mode, benchmark):
    # Extract board size and win condition from the model name
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
    best_policy = PolicyValueNet(width, height, model_file)
    fpa_player = Fictitious_Agent(
        policy_value_function=best_policy.policy_value_fn,
        self_play=False,
        simulations=20,
        depth=TRAINING_PARAMETERS[get_fpa_param_key(width, height, n_in_row)]['depth'],
        action_sample_count=TRAINING_PARAMETERS[get_fpa_param_key(width, height, n_in_row)]['action_sample_count']
    )
    benchmark_player = get_benchmark_player(benchmark)

    # Determine mode of operation: demonstration or rating
    if mode == "dem":
        print(f"Demonstrating a game between {model_name} and {benchmark}.")
        winner = game.start_play(benchmark_player, fpa_player, start_player=0, is_shown=1)
        print("Game over. Winner:", benchmark if winner == benchmark_player.player else "Model" if winner == fpa_player.player else "Tie")

    elif mode == "rate":
        print(f"Rating {model_name} against {benchmark} over 100 games.")
        win_count = 0
        for _ in range(100):
            winner = game.start_play(benchmark_player, fpa_player, start_player=random.randint(0, 1), is_shown=0)
            if winner == fpa_player.player:
                win_count += 1
        win_rate = win_count / 100 * 100
        print(f"Win rate of {model_name} against {benchmark}: {win_rate}%")

    else:
        raise ValueError("Invalid mode. Use 'dem' for demonstration or 'rate' for rating.")

def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, required=True, help="Path to the model file")
    parser.add_argument("--benchmark_model", type=str, default=BenchmarkModel.Random.name, help="Benchmark model: Random or PureMcts")
    parser.add_argument("--mode", type=str, default="rate", help="Mode of operation: 'rate' or 'dem'")
    return parser

if __name__ == "__main__":
    args = get_args().parse_args()
    main(args.model_file, args.mode, BenchmarkModel[args.benchmark_model])
