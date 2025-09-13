import unittest
import numpy as np
from c4 import Board
import torch

class TestWinConditions(unittest.TestCase):

    def test_horizontal(self):
        board = Board()
        board.make_move(0)
        board.make_move(0)
        board.make_move(1)
        board.make_move(1)
        board.make_move(2)
        board.make_move(2)
        board.make_move(3)
        self.assertTrue(board.horizontal_win())

    def test_vertical(self):
        board = Board()
        board.make_move(0)
        board.make_move(1)
        board.make_move(0)
        board.make_move(1)
        board.make_move(0)
        board.make_move(1)
        board.make_move(0)
        self.assertTrue(board.vertical_win())

    def test_diagonal(self):
        board = Board()
        board.make_move(0)
        board.make_move(1)
        board.make_move(1)
        board.make_move(2)
        board.make_move(3)
        board.make_move(2)
        board.make_move(2)
        board.make_move(3)
        board.make_move(3)
        board.make_move(4)
        board.make_move(3)
        self.assertTrue(board.diagonal_win())

    def test_false(self):
        board = Board()
        board.make_move(0)
        board.make_move(1)
        board.make_move(1)
        board.make_move(0)
        board.make_move(0)
        board.make_move(1)
        board.make_move(1)
        self.assertFalse(board.diagonal_win())

    def test_mcts(self):
        policy_model = lambda x: torch.tensor([0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14])
        value_model = lambda x: 0.5
        board = Board()
        board.make_move(3)
        board.make_move(3)
        board.make_move(4)
        board.make_move(4)
        board.make_move(5)

        rewards = [board.get_mcts_reward(policy_model, value_model, board, lookahead) for lookahead in [0,1,2]]
        print(rewards)


if __name__ == "__main__":
    unittest.main()
