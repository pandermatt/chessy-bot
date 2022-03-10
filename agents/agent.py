from chess_env import ChessEnv


class Agent:
    NAME = ''

    def __init__(self):
        self.env = ChessEnv(4)
        print(f"===== {self.NAME} =====")

    def run(self, callback=lambda *args: None):
        pass
