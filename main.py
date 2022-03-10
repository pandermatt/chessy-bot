from agents.chessy_agent import ChessyAgent
from agents.first_five_agent import FirstFiveAgent
from agents.random_agent import RandomAgent

if __name__ == '__main__':
    board_size = 4

    FirstFiveAgent(board_size).run()
    RandomAgent(board_size).run()
    ChessyAgent(board_size).run()
