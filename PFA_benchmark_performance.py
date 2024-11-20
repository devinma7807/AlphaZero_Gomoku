import argparse
import re
import random

from Fictitious_Agent import Fictitious_Agent, TRAINING_PARAMETERS, get_fpa_param_key
from game import Board, Game
from policy_value_net_pytorch import PolicyValueNet
from enum import Enum

from random_vs_model import RandomPlayer
from heuristic_vs_model import HeuristicPlayer
from mcts_pure import MCTSPlayer as MCTS_Pure

FPA_SIMULATIONS = 50
FPA_DEPTHS = 3

class BenchmarkModel(Enum):
    Random = 1
    Heuristic = 2
    PureMcts = 3

    def __str__(self):
        return self.name

def get_benchmark_player(benchmark):
    if benchmark == BenchmarkModel.Random:
        benchmark_player = RandomPlayer()
    elif benchmark == BenchmarkModel.Heuristic:
        benchmark_player = HeuristicPlayer()
    elif benchmark == BenchmarkModel.PureMcts:
        benchmark_player = MCTS_Pure(c_puct=5, n_playout=100)
    else:
        raise ValueError("Benchmark Model Needs to be Initialized")
    return benchmark_player

def main(model_name, mode, benchmark):
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
    best_policy = PolicyValueNet(width, height, model_file)
    fpa_player = Fictitious_Agent(policy_value_function=best_policy.policy_value_fn,
                                  self_play=False,
                                  simulations=FPA_SIMULATIONS,
                                  depth=FPA_DEPTHS,
                                  action_sample_count=TRAINING_PARAMETERS[get_fpa_param_key(width, height, n_in_row)]['action_sample_count'])
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, default = r"FPA_Outputs/best_policy_FPA450_6_6_4")
    parser.add_argument("--benchmark_model", type=str, default=BenchmarkModel.PureMcts.name, help='Needs to be part of setup in BenchmarkModel')
    parser.add_argument('--mode', type=str, default = 'rate', help = 'rate or dem')
    return parser

if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()
    main(args.model_file, args.mode, BenchmarkModel[args.benchmark_model])
