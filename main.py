import numpy as np
import pandas
from matplotlib import pyplot as plt

from agents.q_table_agent import QTableAgent
from config import config
from neuronal_engine.helper import AIM
from util.logger import log
from util.storage_io import dump_file, load_file, is_model_present

intern_output_nr = 500


def savefig(filename):
    plt.tight_layout()
    plt.savefig(config.model_data_file(filename), transparent="True", pad_inches=0)


def print_to_console(nn, _, n, N_episodes, R_save, N_moves_save):
    if n % intern_output_nr == 0 and n > 0:
        dump_file(nn, nn._name)
        log.info(f"Epoche ({n}/{N_episodes})")
        log.info(f"{nn._name}, Average reward: {np.mean(R_save[(n - intern_output_nr):n])} "
                 + f"Number of steps: {np.mean(N_moves_save[(n - intern_output_nr):n])}")


def plot_curve(names, value):
    for idx in range(len(names)):
        plt.plot([i + 1 for i in range(len(value[idx]))], value[idx], label=names[idx])
    plt.legend()
    plt.grid(True)


def plot_curve_avg(names, value, last=100, suffix=""):
    for idx in range(len(names)):
        plt.plot([i + 1 for i in range(last, len(value[idx]))], generate_moving_avg(value[idx], last=last),
                 label=f"{names[idx]}{suffix}")
    plt.legend()
    plt.grid(True)


def print_stats(names, r_saves, step_saves):
    plt.subplots_adjust(wspace=1, hspace=0.3)

    plt.subplot(2, 1, 1)
    plot_curve(names, r_saves)
    plt.title(f"Avg. Rewards")

    plt.subplot(2, 1, 2)
    plot_curve(names, step_saves)
    plt.title(f"Avg. Steps")
    savefig(f"all-learning_curve.png")

    plt.figure()
    plt.subplot(2, 1, 1)
    plot_curve_avg(names, r_saves, suffix=" -- last 100")
    plot_curve_avg(names, r_saves, last=500, suffix=" -- last 500")
    plt.title(f"Avg. Rewards")

    plt.subplot(2, 1, 2)
    plot_curve_avg(names, step_saves, suffix=" -- last 100")
    plot_curve_avg(names, step_saves, last=500, suffix=" -- last 500")
    plt.title(f"Avg. Steps")
    savefig(f"all-learning_curve-100_500_avg.png")

    plt.figure()
    plt.subplot(2, 1, 1)
    plot_curve_avg(names, r_saves, last=500)
    plt.title(f"Avg. Rewards -- last 500")

    plt.subplot(2, 1, 2)
    plot_curve_avg(names, step_saves, last=1000)
    plt.title(f"Avg. Steps -- last 100")
    savefig(f"all-learning_curve-clean.png")


def evaluate_agent(name, reward):
    plt.figure()
    plt.plot([i + 1 for i in range(500, len(reward))], generate_moving_avg(reward, last=500), label="Average last 500")
    plt.plot([i + 1 for i in range(1000, len(reward))], generate_moving_avg(reward, last=1000),
             label="Average last 1000")

    avg = np.average(reward)
    x = np.linspace(0, len(reward))
    plt.plot(x, [avg] * len(x), label="Average")
    plt.plot(x, [AIM] * len(x), label="Aim")
    plt.grid(True)
    plt.legend()
    savefig(f"{name}-learning_curve.png")

    plt.figure()
    plt.plot([i + 1 for i in range(1000, len(reward))], generate_moving_avg(reward, last=1000),
             label="Average last 1000")
    plt.plot(x, [avg] * len(x), label="Average")
    plt.plot(x, [AIM] * len(x), label="Aim")
    plt.grid(True)
    plt.legend()
    savefig(f"{name}-learning_curve_clean.png")


def genrate_box_plots(name, reward):
    plt.figure()
    df = pandas.DataFrame({'Reward': reward[-100:]})
    df.boxplot()
    savefig(f"{name}-boxplot_last_100.png")

    plt.figure()
    df = pandas.DataFrame({'Reward': reward, })
    df.boxplot()
    savefig(f"{name}-boxplot_all.png")


def generate_moving_avg(reward, last=100):
    return [np.average(reward[i - last:i]) for i in range(last, len(reward))]


if __name__ == '__main__':
    board_size = 4

    names = []
    rewards = []
    moves = []
    N_episodes = 500000

    plt.rcParams['figure.figsize'] = [10, 5]
    for agent_class in [QTableAgent]:
        # ,
        #           SarsaChessyAgent, QLearningChessyAgent,
        #           DoubleSARSAChessyAgent, DoubleQLearningChessyAgent]:
        agent = agent_class(N_episodes)
        model_filename = f"{agent._name}_model_content"
        if is_model_present(model_filename):
            name, reward, move = load_file(model_filename)
        else:
            name, reward, move = agent.run(print_to_console)
            dump_file([name, reward, move], model_filename)
        log.info(f'Finished with {len(reward)} Epochs')
        log.info('Generating plots...')
        evaluate_agent(name, reward)
        genrate_box_plots(name, reward)
        names.append(name)
        rewards.append(reward)
        moves.append(move)
        plt.close('all')

    print_stats(names, rewards, moves)
