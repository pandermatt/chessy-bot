from chess_env import ChessEnv


class Agent:
    NAME = ''

    def __init__(self, board_size=4):
        self.env = ChessEnv(board_size)
        print(f"===== {self.NAME} =====")

    def run(self, callback=lambda *args: None):
        pass
