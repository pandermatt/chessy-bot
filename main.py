import numpy as np
from matplotlib import pyplot as plt

from agents.chessy_agent import QLearningChessyAgent, DoubleQLearningChessyAgent, \
    DoubleSARSAChessyAgent, SarsaChessyAgent
from util.storage_io import dump_file

intern_output_nr = 100


def print_to_console(nn, _, n, N_episodes, R_save, N_moves_save):
    if n % intern_output_nr == 0 and n > 0:
        dump_file(nn, nn._name)
        print(f"Epoche ({n}/{N_episodes})")

        print(f'{nn._name}, Average reward:', np.mean(R_save[(n - intern_output_nr):n]),
              'Number of steps: ', np.mean(N_moves_save[(n - intern_output_nr):n]))


def print_stats(n_episodes, names, r_saves, step_saves):
    episodes = range(n_episodes)
    plt.subplots_adjust(wspace=1, hspace=0.3)

    count = len(names)

    plt.subplot(2, 1, 1)
    for idx in range(count):
        plt.plot(episodes, r_saves[idx], label=names[idx])
    plt.title(f"Avg. Rewards")
    plt.legend()

    plt.subplot(2, 1, 2)
    for idx in range(count):
        plt.plot(episodes, step_saves[idx], label=names[idx])
    plt.title(f"Avg. Steps ")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    board_size = 4

    # FirstFiveAgent().run()
    # RandomAgent().run()

    N_episodes = 20000  # THE NUMBER OF GAMES TO BE PLAYED
    name1, reward1, moves1 = SarsaChessyAgent(N_episodes).run(print_to_console)
    name2, reward2, moves2 = QLearningChessyAgent(N_episodes).run(print_to_console)
    name3, reward3, moves3 = DoubleSARSAChessyAgent(N_episodes).run(print_to_console)
    name4, reward4, moves4 = DoubleQLearningChessyAgent(N_episodes).run(print_to_console)

    names = [name1, name2, name3, name4]
    rewards = [reward1, reward2, reward3, reward4]
    moves = [moves1, moves2, moves3, moves4]
    print_stats(N_episodes, names, rewards, moves)
