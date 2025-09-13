from agent import PpoAgent
from c4 import Board


def train(agent: PpoAgent, episodes: int):
    for episode in range(episodes):
        board = Board()
        while not (board.game_won() or board.game_tied()):
            agent.act(board)


agent = PpoAgent((3, 6, 7), 7)
train(agent, 64000000)
