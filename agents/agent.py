from chess_env import ChessEnv
from util.logger import log


class Agent:
    NAME = ''

    def __init__(self):
        self.env = ChessEnv(4)
        log.info(f"===== {self.NAME} =====")
        log.info("Starting...")

    def run(self, callback=lambda *args: None):
        pass
