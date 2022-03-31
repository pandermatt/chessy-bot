from slugify import slugify

from chess_env import ChessEnv
from util.logger import log


class Agent:
    """
    Super Agent class.
    This class manages the chess environment
    and prints information about the agent.
    """
    NAME = ''

    def __init__(self):
        self.env = ChessEnv(4)
        log.info(f"===== {self.NAME} =====")

    def run(self, callback=lambda *args: None):
        """
        Runs the agent.

        @param callback: Method to call after each episode.
        """
        log.info("Starting...")
        log.info(f"...reward_step={self.env.reward_step}")
        log.info(f"...reward_draw={self.env.reward_draw}")
        log.info(f"...reward_checkmate={self.env.reward_checkmate}")

    def clean_name(self):
        return slugify(self.NAME)
