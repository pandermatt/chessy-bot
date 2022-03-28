from slugify import slugify

from chess_env import ChessEnv
from util.logger import log


class Agent:
    NAME = ''

    def __init__(self):
        self.env = ChessEnv(4)
        log.info(f"===== {self.NAME} =====")

    def run(self, callback=lambda *args: None):
        log.info("Starting...")
        log.info(f"...reward_step={self.env.reward_step}")
        log.info(f"...reward_draw={self.env.reward_draw}")
        log.info(f"...reward_checkmate={self.env.reward_checkmate}")
        pass

    def clean_name(self):
        return slugify(self.NAME)
