import argparse
import re
import random
import time  # Import the time module

from game import Board, Game
from random_vs_model import RandomPlayer
from heuristic_vs_model import HeuristicPlayer

def main(mode, dimensions):
    # Start timer
    start_time = time.time()
    
    # Parse dimensions (e.g., "6_6_4" or "8_8_5")
    match = re.match(r"(\d+)_(\d+)_(\d+)", dimensions)
    if match:
        width, height, n_in_row = map(int, match.groups())
    else:
        raise ValueError("Dimensions must be in the format 'WxH_N', e.g., '6_6_4'.")

    # Initialize the board and game
    board = Board(width=width, height=height, n_in_row=n_in_row)
    game = Game(board)

    playerRand = RandomPlayer()
    playerHeu = HeuristicPlayer()

    # Determine mode of operation: demonstration or rating
    if mode == "dem":
        print(f"Demonstrating a game between heuristic model and random model.")
        winner = game.start_play(playerRand, playerHeu, start_player=0, is_shown=1)
        print("Game over. Winner:", "random model" if winner == playerRand.player else "heuristic model" if winner == playerHeu.player else "Tie")

    elif mode == "rate":
        print(f"Rating heuristic model against random model over 20 games.")
        win_count = 0
        for _ in range(20):
            winner = game.start_play(playerRand, playerHeu, start_player=random.randint(0, 1), is_shown=0)
            if winner == playerHeu.player:
                win_count += 1
        win_rate = win_count / 20 * 100
        print(f"Win rate of heuristic model against random model: {win_rate}%")

    else:
        raise ValueError("Invalid mode. Use 'dem' for demonstration or 'rate' for rating.")
    
    # End timer and print elapsed time
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a game demonstration or rate a model.")
    parser.add_argument(
        "mode",
        type=str,
        choices=["dem", "rate"],
        help="Mode of operation: 'dem' for demonstration, 'rate' for rating"
    )
    parser.add_argument(
        "dimensions",
        type=str,
        help="Board dimensions and win condition in the format 'WxH_N', e.g., '6_6_4'"
    )
    args = parser.parse_args()
    main(args.mode, args.dimensions)
