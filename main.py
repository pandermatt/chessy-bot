from matplotlib import pyplot as plt

from agents.chessy_agent import QLearningChessyAgent, SarsaChessyAgent
from agents.first_five_agent import FirstFiveAgent
from agents.random_agent import RandomAgent


def print_stats(n_episodes, name1, r_save1, step_save1, name2, r_save2, step_save2):
    episodes = range(n_episodes)
    plt.subplots_adjust(wspace=1, hspace=0.3)

    plt.subplot(2, 1, 1)
    plt.plot(episodes, r_save1, label=name1)
    plt.plot(episodes, r_save2, label=name2)
    plt.title(f"Avg. Rewards")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(episodes, step_save1, label=name1)
    plt.plot(episodes, step_save2, label=name2)
    plt.title(f"Avg. Steps ")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    board_size = 4

    FirstFiveAgent().run()
    RandomAgent().run()

    N_episodes = 1000  # THE NUMBER OF GAMES TO BE PLAYED
    name1, reward1, moves1 = SarsaChessyAgent(N_episodes).run()
    name2, reward2, moves2 = QLearningChessyAgent(N_episodes).run()
    print_stats(N_episodes, name1, reward1, moves1, name2, reward2, moves2)
