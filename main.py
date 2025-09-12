#!/usr/bin/env python3
"""
Main entry point for the Number Guessing Game
A simple console-based game using Python standard library
"""

from c4 import C4, GPTPlayer, PPOPlayer, RandomPlayer, TerminalPlayer
from utils import get_user_choice, clear_screen


def main():
    """Main game loop"""
    player0_name = input("Enter name player 1 (enter ai for AI player):")
    player0 = TerminalPlayer(
        player0_name) if player0_name != "ai" else PPOPlayer("AI-1")

    player1_name = input("Enter name player 2 (enter ai for AI player):")
    player1 = TerminalPlayer(
        player1_name) if player1_name != "ai" else PPOPlayer("AI-2")

    game = C4(player0, player1)

    while True:
        while not game.done:
            clear_screen()

            print(game.board)
            game.play_round()

            if game.done:
                clear_screen()
                print(game.board)
                print(f"{game.winner.name if game.winner else 'Nobody'} wins")

        play_again = get_user_choice("\nWould you like to play again? (y/n): ",
                                     valid_choices=['y', 'yes', 'n', 'no'])

        if play_again.lower() in ['n', 'no']:
            break

        clear_screen()
        print("\n" + "=" * 50)
        print("Starting a new game...")
        print("=" * 50)
        game.reset()


if __name__ == "__main__":
    main()
