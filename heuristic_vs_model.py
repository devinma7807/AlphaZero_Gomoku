# heuristic_vs_model.py
import re
import random
import numpy as np
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy as TheanoPolicyValueNet
from policy_value_net_pytorch_ResNet import PolicyValueNet as PytorchPolicyValueNet
import pickle
import torch
import os
from tqdm import tqdm
import sys
import time  # Import the time module

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
            # move_scores[move] = self.evaluate_move(board, move)
            # Check for critical blocking rules first
            if self.blocks_opponent_critical(board, move):
                move_scores[move] = float('inf')
            
            # Otherwise, calculate heuristic score
            else:
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
        # Save the current state of the board
        original_states = board.states.copy()
        original_availables = board.availables[:]
        original_current_player = board.current_player

        score = 0
        try:
            # Simulate the move
            board.do_move(move)

            # Evaluate the sequence after the move
            score += self.evaluate_sequence(board, move, self.player)
        finally:
            # Restore the original state of the board
            board.states = original_states
            board.availables = original_availables
            board.current_player = original_current_player

        return score


    def evaluate_sequence(self, board, move, player):
        """Evaluate the score based on created sequences."""
        score = 0
        target = board.n_in_row
        # Check all directions (horizontal, vertical, two diagonals)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for direction in directions:
            line_length,_ = self.count_in_a_row(board, move, player, direction)
            if line_length >= target:
                score += 100  # Winning move
            else:
                for i in range(2, target):
                    if line_length == i:
                        score += i
                        break  # No need to check further in this direction
        return score

    def count_in_a_row(self, board, move, player, direction):
        """Count consecutive stones of the given player in both directions from 'move'."""
        dx, dy = direction
        x, y = move // board.width, move % board.width  # Corrected line for row and column extraction
        
        count = 1  # Start with the current position as part of the sequence
        open_ends = 0

        if x == 0 or y == 0 or x == board.width - 1 or y == board.height - 1:
            open_ends += 1 

        # Check forward direction
        forward_count = 0
        fx, fy = x + dx, y + dy
        while 0 <= fx < board.width and 0 <= fy < board.height:
            forward_pos = fx * board.width + fy
            if board.states.get(forward_pos) == player:
                forward_count += 1
                fx += dx
                fy += dy
            elif board.states.get(forward_pos) is None:
                open_ends += 1
                break
            else:
                break  # Stop if we reach an empty spot or opponent's stone
        
        # Check backward direction
        backward_count = 0
        bx, by = x - dx, y - dy
        while 0 <= bx < board.width and 0 <= by < board.height:
            backward_pos = bx * board.width + by
            if board.states.get(backward_pos) == player:
                backward_count += 1
                bx -= dx
                by -= dy
            elif board.states.get(backward_pos) is None:
                open_ends += 1
                break
            else:
                break  # Stop if we reach an empty spot or opponent's stone

        # Total consecutive count including the 'move' position
        count = 1 + forward_count + backward_count
        return count, open_ends


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
        cnt, openEnd = self.count_in_a_row(board, move, self.opponent, direction)
        return cnt >= board.n_in_row - 1 and openEnd == 2


    def blocks_four_one_open(self, board, move, direction):
        """Check if the move blocks an opponent's 4-in-a-row with one open end."""
        cnt, openEnd = self.count_in_a_row(board, move, self.opponent, direction)
        return cnt >= board.n_in_row and openEnd == 1


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
            try:
                # Attempt to load as a pickle file
                os.environ['THEANO_FLAGS'] = 'device=cuda,floatX=float32'
                policy_param = pickle.load(f, encoding='latin1')
                print("Loaded model file as a pickle.")
                best_policy = TheanoPolicyValueNet(width, height, policy_param)
            except pickle.UnpicklingError:
                # If pickle fails, try loading as a PyTorch model
                f.seek(0)  # Reset file pointer to the beginning
                policy_param = torch.load(f, map_location=torch.device('cuda'), weights_only=True)  # Load PyTorch model
                print("Loaded model file as a PyTorch model.")
                best_policy = PytorchPolicyValueNet(width, height)
                best_policy.policy_value_net.load_state_dict(policy_param)
                print(type(policy_param))  # Should output <class 'dict'> for state_dict
                if isinstance(policy_param, dict):
                    print("Keys in policy_param (state_dict):", policy_param.keys())
                else:
                    print("Loaded policy_param is not a state_dict. Check your model file format.")

    except FileNotFoundError:
        raise ValueError(f"Model file '{model_file}' not found.")

    
    mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

    # Initialize the heuristic player
    heuristic_player = HeuristicPlayer()

    # Determine mode of operation: demonstration or rating
    if mode == "dem":
        print(f"Demonstrating a game between {model_name} and HeuristicPlayer.")
        winner = game.start_play(heuristic_player, mcts_player, start_player=0, is_shown=1)
        print("The model name for Winner is:", "Heuristic" if winner == heuristic_player.player else model_name if winner == mcts_player.player else "Tie")

    elif mode == "rate":
        num_games = 10
        print(f"Rating {model_name} against HeuristicPlayer over {num_games} games.")
        win_count = 0
        for _ in tqdm(range(num_games), desc="Progress", unit="game"):
            winner = game.start_play(heuristic_player, mcts_player, start_player=random.randint(0, 1), is_shown=0)
            if winner == mcts_player.player:
                win_count += 1
        win_rate = win_count / num_games * 100
        print(f"Win rate of {model_name} against HeuristicPlayer: {win_rate}%")

    else:
        raise ValueError("Invalid mode. Use 'dem' for demonstration or 'rate' for rating.")

if __name__ == "__main__":
    # Example usage:
    # main("best_policy_8_8_5", "dem") for demonstration
    # main("best_policy_8_8_5", "rate") for rating
    # main("best_policy_8_8_5", "dem")
    # main("best_policy_6_6_4_ResNet", "dem")
    if len(sys.argv) != 3:
        print("Usage: python heuristic_vs_model.py <model_name> <mode>")
        print("Example: python heuristic_vs_model.py best_policy_8_8_5 dem")
    else:
        start_time = time.time()
        main(sys.argv[1], sys.argv[2])
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Total running time: {elapsed_time:.2f} seconds")
# #    Set board parameters
#     width, height = 6, 6
#     n_in_row = 4  # Winning condition: 4 in a row
#     board = Board(width=width, height=height, n_in_row=n_in_row)

#     # Set up the board state
#     board.states = {
#         21: 1,  # X at (3, 3)
#         13: 1,  # X at (2, 1)
#         9: 1,   # X at (1, 3)
#         20: 1,  # X at (3, 2)
#         14: 2,  # O at (2, 2)
#         15: 2,  # O at (2, 3)
#         19: 2,  # O at (3, 1)
#         16: 2,  # O at (2, 4)
#     }
#     board.availables = [0,1,2,3,4,5,6,7,8,10,11,12,17,18,22,23,24,25,26,27,28,29,30,31,32,33,34,35]  # Set available moves to include potential blocking positions
#     board.current_player = 1  # X to play

#     # Initialize HeuristicPlayer
#     player = HeuristicPlayer()
#     player.set_player_ind(1)  # Set as player 1 (X)

#     # Check if blocks_opponent_critical correctly identifies the critical moves
#     critical_positions = [0,1,2,3,4,5,6,7,8,10,11,12,17,18,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
#     for move in critical_positions:
#         is_critical = player.blocks_opponent_critical(board, move)
#         print(f"Move {move} at position ({move // width}, {move % width}) is critical: {is_critical}")

#     # Test the action selection
#     move = player.get_action(board)
#     print(f"Player X selected move: {move} (expected one of {critical_positions})")
#     assert move in critical_positions, "Player X failed to block a critical move."
