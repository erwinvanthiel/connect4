"""
Utility functions for the Number Guessing Game
Contains helper functions for input validation and display formatting
"""

import os
import sys


def clear_screen():
    """Clear the console screen for better user experience"""
    os.system('cls' if os.name == 'nt' else 'clear')


def display_welcome_message():
    """Display the game's welcome message and instructions"""
    clear_screen()
    print("=" * 60)
    print("🎯 WELCOME TO FOUR IN A ROW! 🎯")
    print("=" * 60)
    input("Press Enter to start playing... 🚀")


def get_valid_integer(prompt, min_value=None, max_value=None):
    """
    Get a valid integer input from the user with optional range validation
    
    Args:
        prompt (str): The prompt message to display
        min_value (int, optional): Minimum allowed value
        max_value (int, optional): Maximum allowed value
    
    Returns:
        int: Valid integer input from user
    """
    while True:
        try:
            user_input = input(prompt).strip()

            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Thanks for playing! Goodbye!")
                sys.exit(0)

            value = int(user_input)

            # Validate range if specified
            if min_value is not None and value < min_value:
                print(f"❌ Please enter a number >= {min_value}")
                continue

            if max_value is not None and value > max_value:
                print(f"❌ Please enter a number <= {max_value}")
                continue

            return value

        except ValueError:
            print("❌ Please enter a valid number!")
        except KeyboardInterrupt:
            print("\n👋 Thanks for playing! Goodbye!")
            sys.exit(0)


def get_user_choice(prompt, valid_choices):
    """
    Get a user choice from a list of valid options
    
    Args:
        prompt (str): The prompt message to display
        valid_choices (list): List of valid choice strings
    
    Returns:
        str: Valid user choice (lowercase)
    """
    while True:
        try:
            choice = input(prompt).strip().lower()

            # Check for exit commands
            if choice in ['quit', 'exit', 'q']:
                print("👋 Thanks for playing! Goodbye!")
                sys.exit(0)

            if choice in [str(option).lower() for option in valid_choices]:
                return choice

            print(
                f"❌ Please enter one of: {', '.join([str(c) for c in valid_choices])}"
            )

        except KeyboardInterrupt:
            print("\n👋 Thanks for playing! Goodbye!")
            sys.exit(0)
