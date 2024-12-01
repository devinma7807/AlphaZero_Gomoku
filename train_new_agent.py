# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function

import datetime
import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from Fictitious_Agent import Fictitious_Agent, TRAINING_PARAMETERS, get_fpa_param_key
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet # Keras

class TrainPipeline():
    def __init__(self, init_model=None):
        # params of the board and the game
        self.board_width = 8
        self.board_height = 8
        self.n_in_row = 5

        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        self.board_type =get_fpa_param_key(self.board_width, self.board_height, self.n_in_row)

        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.buffer_size = 10000
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02

        # Params to change
        self.n_playout = TRAINING_PARAMETERS[self.board_type]['n_playout'] # 400
        self.game_batch_num = TRAINING_PARAMETERS[self.board_type]['game_batch_num'] #1500
        self.batch_size = TRAINING_PARAMETERS[self.board_type]['batch_size'] # 512
        self.check_freq = TRAINING_PARAMETERS[self.board_type]['check_freq']  # 50
        self.save_freq = TRAINING_PARAMETERS[self.board_type]['save_freq']
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = TRAINING_PARAMETERS[self.board_type]['pure_mcts_playout_num']
        self.pure_mcts_step = TRAINING_PARAMETERS[self.board_type]['pure_mcts_step'] #1000
        self.pure_mcts_max_num = TRAINING_PARAMETERS[self.board_type]['pure_mcts_max_num'] #5000
        self.depth = TRAINING_PARAMETERS[self.board_type]['depth']

        # Parameters used for evaluation with pure mcts
        self.c_puct = 5
        self.best_win_ratio = 0.0

        # initialize network models
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)

        self.fictitious_agent = Fictitious_Agent(policy_value_function=self.policy_value_net.policy_value_fn,
                                                 self_play=True,
                                                 simulations=self.n_playout,
                                                 depth=self.depth,
                                                 action_sample_count=TRAINING_PARAMETERS[self.board_type]['action_sample_count'])


    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, agent_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, agent_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_agent_prob = np.rot90(np.flipud(
                    agent_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_agent_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_agent_prob = np.fliplr(equi_agent_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_agent_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.fictitious_agent,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)


    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        agent_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    agent_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_fictitious_player = Fictitious_Agent(policy_value_function=self.policy_value_net.policy_value_fn,
                                                     self_play=False,
                                                     simulations=self.n_playout,
                                                     depth=self.depth,
                                                     action_sample_count=TRAINING_PARAMETERS[self.board_type]['action_sample_count'])

        pure_mcts_player = MCTS_Pure(c_puct=self.c_puct,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_fictitious_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                print(f"Start Batch {i} at time {datetime.datetime.now()}")
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                    print(f"Trained game batch {i} at time {datetime.datetime.now()}")
                    if i % self.save_freq == 0:
                        self.policy_value_net.save_model('./FPA_Outputs/current_policy_FPA_8_8_5_6_sample_4_depth.model')
                        print(f"Saved Model at time {datetime.datetime.now()}")
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch used to evaluate agent: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    print(f"Win Ratio: {win_ratio}")
                    self.policy_value_net.save_model('./FPA_Outputs/current_policy_FPA_8_8_5_6_sample_4_depth.model')
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('./FPA_Outputs/best_policy_FPA_8_8_5_6_sample_4_depth.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < self.pure_mcts_max_num):
                            self.pure_mcts_playout_num += self.pure_mcts_step
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')

if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
