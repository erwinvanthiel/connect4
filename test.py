import unittest
import numpy as np
from c4 import Board


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


if __name__ == "__main__":
    unittest.main()
