# heuristic_vs_model.py
import re
import random
import numpy as np
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy
import pickle

class HeuristicPlayer:
    """A heuristic-based player, structured similarly to MCTSPlayer."""

    def __init__(self):
        self.player = None
        self.is_first_move = True

    def set_player_ind(self, p):
        self.player = p
        self.opponent = 1 if p == 2 else 2

    def reset_player(self):
        self.is_first_move = True  # Reset for a new game

    def get_action(self, board, temp=1e-3, return_prob=0):
        """Select a move based on heuristic evaluation."""
        sensible_moves = board.availables
        if not sensible_moves:
            print("No moves available.")
            return None

        # First-move rule: try to place in the center or nearest to center
        if self.is_first_move:
            self.is_first_move = False  # Toggle off first-move status
            center_move = self.get_center_move(board, sensible_moves)
            if center_move is not None:
                move_probs = np.zeros(board.width * board.height)
                move_probs[center_move] = 1  # 100% probability on center move
                return (center_move, move_probs) if return_prob else center_move

        # Evaluate each move based on heuristics
        move_scores = {}
        for move in sensible_moves:
            # Check for critical blocking rules first
            if self.blocks_opponent_critical(board, move):
                return move  # Directly return the move if it satisfies critical blocking
            
            # Otherwise, calculate heuristic score
            move_scores[move] = self.evaluate_move(board, move)

        # Select the move with the highest heuristic score
        best_move = max(move_scores, key=move_scores.get)
        move_probs = np.zeros(board.width * board.height)
        move_probs[best_move] = 1  # 100% probability on the chosen move
        return (best_move, move_probs) if return_prob else best_move

    def get_center_move(self, board, sensible_moves):
        """Get the center move or the closest available move to the center."""
        center_x, center_y = board.width // 2, board.height // 2
        center_move = center_y * board.width + center_x
        if center_move in sensible_moves:
            return center_move  # If the center is available, take it

        # If the center is not available, find the closest move
        def distance_to_center(move):
            x, y = move % board.width, move // board.width
            return (x - center_x) ** 2 + (y - center_y) ** 2

        closest_move = min(sensible_moves, key=distance_to_center)
        return closest_move

    def evaluate_move(self, board, move):
        """Evaluate the heuristic score of a move."""
        score = 0
        # Add the move to simulate its effect
        board.do_move(move)
        
        # Own sequence evaluation
        score += self.evaluate_sequence(board, move, self.player)

        # Undo move after evaluation
        board.states.pop(move)
        board.availables.append(move)
        return score

    def evaluate_sequence(self, board, move, player):
        """Evaluate the score based on created sequences."""
        score = 0
        # Check all directions (horizontal, vertical, two diagonals)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for direction in directions:
            line_length = self.count_in_a_row(board, move, player, direction)
            if line_length == 2:
                score += 2
            elif line_length == 3:
                score += 3
            elif line_length == 4:
                score += 4
            elif line_length >= 5:
                score += 100  # Winning move
        return score

    def count_in_a_row(self, board, move, player, direction):
        """Count consecutive stones in a row in the given direction."""
        dx, dy = direction
        count = 1  # Start with the current move
        for d in [1, -1]:  # Check both forward and backward
            x, y = move % board.width, move // board.width
            while True:
                x, y = x + dx * d, y + dy * d
                if 0 <= x < board.width and 0 <= y < board.height:
                    pos = y * board.width + x
                    if board.states.get(pos) == player:
                        count += 1
                    else:
                        break
                else:
                    break
        return count

    def blocks_opponent_critical(self, board, move):
        """Check if the move blocks an opponent's critical sequence."""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for direction in directions:
            # Check for blocking scenarios based on opponent's sequences
            if self.blocks_three_open(board, move, direction) or self.blocks_four_one_open(board, move, direction):
                return True
        return False

    def blocks_three_open(self, board, move, direction):
        """Check if the move blocks an opponent's 3-in-a-row with both ends open."""
        return self.count_in_a_row(board, move, self.opponent, direction) >= 3

    def blocks_four_one_open(self, board, move, direction):
        """Check if the move blocks an opponent's 4-in-a-row with one open end."""
        return self.count_in_a_row(board, move, self.opponent, direction) >= 4

    def __str__(self):
        return "HeuristicPlayer {}".format(self.player)

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

    # Initialize the heuristic player
    heuristic_player = HeuristicPlayer()

    # Determine mode of operation: demonstration or rating
    if mode == "dem":
        print(f"Demonstrating a game between {model_name} and HeuristicPlayer.")
        winner = game.start_play(heuristic_player, mcts_player, start_player=0, is_shown=1)
        print("Game over. Winner:", "Heuristic" if winner == heuristic_player.player else "Model" if winner == mcts_player.player else "Tie")

    elif mode == "rate":
        print(f"Rating {model_name} against HeuristicPlayer over 100 games.")
        win_count = 0
        for _ in range(100):
            winner = game.start_play(heuristic_player, mcts_player, start_player=random.randint(0, 1), is_shown=0)
            if winner == mcts_player.player:
                win_count += 1
        win_rate = win_count / 100 * 100
        print(f"Win rate of {model_name} against HeuristicPlayer: {win_rate}%")

    else:
        raise ValueError("Invalid mode. Use 'dem' for demonstration or 'rate' for rating.")

if __name__ == "__main__":
    # Example usage:
    # main("best_policy_8_8_5", "dem") for demonstration
    # main("best_policy_8_8_5", "rate") for rating
    import sys
    if len(sys.argv) != 3:
        print("Usage: python heuristic_vs_model.py <model_name> <mode>")
        print("Example: python heuristic_vs_model.py best_policy_8_8_5 dem")
    else:
        main(sys.argv[1], sys.argv[2])
