import numpy as np
import pandas
from matplotlib import pyplot as plt

from agents.chessy_agent import QLearningChessyAgent, DoubleQLearningChessyAgent, \
    DoubleSARSAChessyAgent, SarsaChessyAgent
from config import config
from util.logger import log
from util.storage_io import dump_file

intern_output_nr = 500


def print_to_console(nn, _, n, N_episodes, R_save, N_moves_save):
    if n % intern_output_nr == 0 and n > 0:
        dump_file(nn, nn._name)
        log.info(f"Epoche ({n}/{N_episodes})")
        log.info(f"{nn._name}, Average reward: {np.mean(R_save[(n - intern_output_nr):n])} "
                 + f"Number of steps: {np.mean(N_moves_save[(n - intern_output_nr):n])}")


def plot_curve(episodes, names, value):
    for idx in range(len(names)):
        plt.plot([i + 1 for i in range(100, len(value[idx]))], generate_moving_avg(value[idx]),
                 label=f"{names[idx]} - avg last 100")
        plt.plot(episodes, value[idx], label=names[idx])
    plt.legend()


def print_stats(n_episodes, names, r_saves, step_saves):
    episodes = range(n_episodes)
    plt.subplots_adjust(wspace=1, hspace=0.3)

    plt.subplot(2, 1, 1)
    plot_curve(episodes, names, r_saves)
    plt.title(f"Avg. Rewards")

    plt.subplot(2, 1, 2)
    plot_curve(episodes, names, step_saves)
    plt.title(f"Avg. Steps ")

    plt.show()


def evaluate_agent(name, reward):
    plt.figure()
    plt.plot([i + 1 for i in range(0, len(reward))], reward, label="Learning Curve")
    plt.plot([i + 1 for i in range(50, len(reward))], generate_moving_avg(reward, last=50), label="Average last 50")

    avg = np.average(reward)
    x = np.linspace(0, len(reward))
    plt.plot(x, [avg] * len(x))
    plt.grid(True)
    plt.legend()
    plt.savefig(config.model_data_file(f"{name}-learning_curve.png"))

    plt.figure()
    plt.plot([i + 1 for i in range(100, len(reward))], generate_moving_avg(reward), label="Average last 100")
    plt.plot(x, [avg] * len(x))
    plt.grid(True)
    plt.legend()
    plt.savefig(config.model_data_file(f"{name}-learning_curve_clean.png"))

    if len(reward) < 100:
        return

    plt.figure()
    plt.plot([i + 1 for i in range(len(reward) - 100, len(reward))], reward[-100:],
             label="Learning Curve")
    x = np.linspace(len(reward) - 100, len(reward))
    plt.plot(x, [avg] * len(x))
    plt.grid(True)
    plt.legend()
    plt.savefig(config.model_data_file(f"{name}-learning_curve_last_100.png"))


def genrate_box_plots(name, reward):
    plt.figure()
    df = pandas.DataFrame({'Reward': reward[-100:]})
    df.boxplot()
    plt.savefig(config.model_data_file(f"{name}-boxplot_last_100.png"))

    plt.figure()
    df = pandas.DataFrame({'Reward': reward, })
    df.boxplot()
    plt.savefig(config.model_data_file(f"{name}-boxplot_all.png"))


def generate_moving_avg(reward, last=100):
    return [np.average(reward[i - last:i]) for i in range(last, len(reward))]


if __name__ == '__main__':
    board_size = 4

    # FirstFiveAgent().run()
    # RandomAgent().run()
    names = []
    rewards = []
    moves = []
    N_episodes = 10000  # THE NUMBER OF GAMES TO BE PLAYED

    for agent in [SarsaChessyAgent, QLearningChessyAgent,
                  DoubleSARSAChessyAgent, DoubleQLearningChessyAgent]:
        name, reward, move = agent(N_episodes).run(print_to_console)
        evaluate_agent(name, reward)
        genrate_box_plots(name, reward)
        names.append(name)
        rewards.append(reward)
        moves.append(move)
    plt.close('all')

    print_stats(N_episodes, names, rewards, moves)
