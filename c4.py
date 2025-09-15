from typing import Tuple, Self
from abc import ABC, abstractmethod
from utils import get_user_choice
import random
import time
import openai
import numpy as np
from dotenv import load_dotenv
from agent import PpoAgent
import torch

class Board:
    """Four in a Row board implementation"""

    def __init__(self, board: torch.Tensor=None, turn: int=0) -> None:
        self.dims: Tuple[int, int] = (6, 7)
        self.board = board if board is not None else torch.zeros((3, self.dims[0], self.dims[1]), dtype=torch.int)
        self.turn = turn

    def get_current_player(self) -> int:
        if self.turn % 2 == 0:
            return 0
        else:
            return 1

    def __str__(self) -> str:
        # Custom 6x5 ASCII art for each symbol

        # ANSI color codes
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"

        # Fully-filled 12x5 ASCII art
        art = [[
            f"{YELLOW} OOOOOOOOOO {RESET}", f"{YELLOW}OO        OO{RESET}",
            f"{YELLOW}OO        OO{RESET}", f"{YELLOW}OO        OO{RESET}",
            f"{YELLOW} OOOOOOOOOO {RESET}"
        ],
               [
                   f"{GREEN} OOOOOOOOOO {RESET}",
                   f"{GREEN}OO        OO{RESET}",
                   f"{GREEN}OO        OO{RESET}",
                   f"{GREEN}OO        OO{RESET}", f"{GREEN} OOOOOOOOOO {RESET}"
               ],
               [
                   "            ", "            ", "            ",
                   "            ", "            "
               ]]

        cell_width = 12
        cell_height = 5

        horizontal_border = "+" + "+".join(
            ["-" * cell_width for _ in range(self.dims[1])]) + "+"

        lines = []
        for row in range(self.dims[0]):
            lines.append(horizontal_border)
            for i in range(cell_height):
                line = "|"
                for col in range(self.dims[1]):
                    cell = self.board[:, row, col]
                    if torch.sum(cell) == 0:
                        player_index = 2  # Empty
                    else:
                        player_index = torch.argmax(cell)  # Player 0 or 1
                    line += art[player_index][i] + "|"
                lines.append(line)
        lines.append(horizontal_border)

        return "\n".join(lines)

    def get_valid_moves(self) -> torch.Tensor:
        """Get a list of valid column indices for the current player"""
        column_sums = torch.sum(self.board[:-1].cpu(), axis=(0, 1))
        return torch.where(column_sums < self.dims[0])[0]

    def make_move(self, move: int, in_place=True) -> Self:
        """Play a round of the game by dropping a piece in the specified column"""
        if move not in self.get_valid_moves():
            return None

        # Find the lowest empty row in the specified column
        row = self.dims[0] - 1
        while row >= 0 and torch.sum(self.board[:-1], axis=0)[row][move] != 0:
            row -= 1

        _board = self if in_place else Board(self.board.clone(), self.turn)
        _board.board[self.get_current_player()][row][move] = 1
        _board.turn += 1
        _board.board[2][:][:] = _board.get_current_player(
        )  # update the turn board

        return _board

    def diagonal_win(self) -> bool:
        """Check if the current player has won the game diagonally"""
        roll_and_add_player0 = self.board[0].clone(
        )  # diagonals bottom left to top right
        roll_and_add_player0_norm = self.board[0].clone()
        roll_and_add_player1 = self.board[1].clone(
        )  # diagonals top left to bottom right
        roll_and_add_player1_norm = self.board[1].clone()

        for i in range(3):
            roll_and_add_player0 += torch.roll(torch.roll(self.board[0],
                                                    shifts=-(i + 1),
                                                    dims=0),
                                               shifts=i + 1,
                                               dims=1)
            roll_and_add_player1 += torch.roll(torch.roll(self.board[1],
                                                          shifts=-(i + 1),
                                                          dims=0),
                                               shifts=i + 1,
                                               dims=1)
            roll_and_add_player0_norm += torch.roll(torch.roll(self.board[0],
                                                               shifts=i + 1,
                                                               dims=0),
                                                    shifts=i + 1,
                                                    dims=1)
            roll_and_add_player1_norm += torch.roll(torch.roll(self.board[1],
                                                               shifts=i + 1,
                                                               dims=0),
                                                    shifts=i + 1,
                                                    dims=1)

        roll_and_add_player0[-3:, :] = 0
        roll_and_add_player0[:, :3] = 0
        roll_and_add_player1[-3:, :] = 0
        roll_and_add_player1[:, :3] = 0

        roll_and_add_player0_norm[:, :3] = 0
        roll_and_add_player0_norm[:3, :] = 0
        roll_and_add_player1_norm[:, :3] = 0
        roll_and_add_player1_norm[:3, :] = 0

        return torch.max(roll_and_add_player0) == 4 or torch.max(
            roll_and_add_player1) == 4 or torch.max(
                roll_and_add_player0_norm) == 4 or torch.max(
                    roll_and_add_player1_norm) == 4

    def horizontal_win(self) -> bool:
        """Check if the current player has won the game horizontally"""
        roll_and_add_player0 = self.board[0].clone()
        roll_and_add_player1 = self.board[1].clone()

        for i in range(3):
            roll_and_add_player0 += torch.roll(self.board[0], shifts=i + 1, dims=1)
            roll_and_add_player1 += torch.roll(self.board[1], shifts=i + 1, dims=1)

        roll_and_add_player0[:, :3] = 0
        roll_and_add_player1[:, :3] = 0
        return torch.max(roll_and_add_player0) == 4 or torch.max(
            roll_and_add_player1) == 4

    def vertical_win(self) -> bool:
        """Check if the current player has won the game vertically"""
        roll_and_add_player0 = self.board[0].clone()
        roll_and_add_player1 = self.board[1].clone()

        for i in range(3):
            roll_and_add_player0 += torch.roll(self.board[0], shifts=i + 1, dims=0)
            roll_and_add_player1 +=torch.roll(self.board[1], shifts=i + 1, dims=0)

        roll_and_add_player0[:3, :] = 0
        roll_and_add_player1[:3, :] = 0
        return torch.max(roll_and_add_player0) == 4 or torch.max(
            roll_and_add_player1) == 4

    def game_won(self) -> bool:
        """Check if the current player has won the game"""
        return self.diagonal_win() or self.horizontal_win(
        ) or self.vertical_win()

    def game_tied(self) -> bool:
        return torch.sum(self.board,
                      axis=(0, 1, 2)) == self.dims[0] * self.dims[1]

    def get_reward(self) -> int:
        if self.game_won():
            return 1
        else:
            return 0

    def get_mcts_reward(self, policy_model, value_model, board, lookahead, current_player=True, device="cpu") -> float:
        if lookahead == 0 or board.get_reward() == 1:
            return (board.get_reward()) * (1 if current_player else -1)
        else:
            valid_moves = board.get_valid_moves()
            possible_boards = [board.make_move(move, in_place=False) for move in valid_moves]
            board.board = board.board.to(device)
            move_probabilities = policy_model(board.board)[valid_moves]
            expected_reward = torch.sum(move_probabilities * torch.tensor([self.get_mcts_reward(policy_model, value_model, board, lookahead - 1, not current_player, device) for board in possible_boards]).to(device))
            return expected_reward


class Player(ABC):
    """Player class for the Four in a Row game"""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def get_move(self, board: Board) -> int:
        pass


class C4:
    """Four in a Row game implementation"""

    def __init__(self, player0: Player, player1: Player) -> None:
        super().__init__()
        self.board: Board = Board()
        self.done: bool = False
        self.winner: Player | None = None
        self.player0: Player = player0
        self.player1: Player = player1

    def reset(self) -> None:
        self.board: Board = Board()
        self.done: bool = False
        self.winner: Player | None = None

    def play_round(self) -> None:
        """Play a round of the game"""
        # get move from current player
        current_player = self.board.get_current_player()
        if current_player == 0:
            self.board.make_move(self.player0.get_move(self.board))
        else:
            self.board.make_move(self.player1.get_move(self.board))

        # check terminal conditions
        if self.board.game_won():
            self.winner = self.player0 if current_player == 0 else self.player1
            self.done = True
            return
        if self.board.game_tied():
            self.done = True

    def get_reward(self, player: Player) -> int:
        if self.winner is None:
            return 0
        else:
            return 2 * int(self.winner == player) - 1


class TerminalPlayer(Player):
    """Terminal player for the Four in a Row game"""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def get_move(self, board: Board) -> int:
        move = get_user_choice(f"\n {self.name} play your move: ",
                               valid_choices=board.get_valid_moves())
        return int(move)


class RandomPlayer(Player):
    """Random player for the Four in a Row game"""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def get_move(self, board: Board) -> int:
        valid_moves = board.get_valid_moves()
        time.sleep(3)
        return valid_moves[random.randint(0, len(valid_moves) - 1)]


class PPOPlayer(Player):
    """RL/PPO player for the Four in a Row game"""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.agent = PpoAgent((3, 6, 7), 7, load_from_path="models/model.pth", train=False)

    def get_move(self, board: Board) -> int:
        a, _, _ = self.agent.choose_action(board)
        time.sleep(0.75)
        return a.item()


class GPTPlayer(Player):
    """GPT player for the Four in a Row game"""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        load_dotenv()
        self.client = openai.OpenAI()

    def get_move(self, board: Board) -> int:

        # Convert board to string format
        def board_to_string(board):
            rows = []
            for r in range(board.dims[0]):
                rows.append("".join("X" if board.board[0][r][c] == 1 else (
                    "O" if board.board[1][r][c] == 1 else "_")
                                    for c in range(board.dims[1])))
            return "\n".join(rows)

        board_str = board_to_string(board)

        # Construct the prompt
        prompt = f"""
    You are playing Connect Four. The board is a {board.dims[0]}-row by {board.dims[1]}-column grid.
    The goal is to get four consecutive pieces horizontally, vertically, or diagonally, with no pieces of the opponent in between.
    The pieces fall to the lowest available row in the chosen column.
    You are player '{"X" if board.get_current_player() == 0 else "O"}'. Your opponent is player '{'X' if board.get_current_player() == 'O' else 'O'}'.
    You get to play the next move. Try to get four in a row horizontally, vertically, or diagonally. Don't let your opponent get four in a row. He will play after you.
    Here is the current board:

    \n{board_str}

    Please choose a column from {','.join([str(x) for x in board.get_valid_moves()])} to drop your piece. Respond with only the column number.
    """
        response = self.client.chat.completions.create(model="gpt-4o-mini",
                                                       messages=[{
                                                           "role":
                                                           "user",
                                                           "content":
                                                           prompt
                                                       }])

        # Extract the response content
        reply = response.choices[0].message.content

        try:
            column = int(reply)
            if column in board.get_valid_moves():
                return column
            else:
                raise ValueError("Invalid value")
        except ValueError:
            valid_moves = board.get_valid_moves()
            return valid_moves[random.randint(0, len(valid_moves) - 1)]
