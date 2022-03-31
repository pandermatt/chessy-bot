import numpy as np
import pandas
from matplotlib import pyplot as plt

from config import config
from neuronal_engine.helper import AIM
from util.logger import log


# HELPER METHODS FOR THE PLOTTING OF THE AGENTS

def print_stats(name, r_save, step_save):
    log.info('------------------------')
    log.info(name)
    log.info(f"Average reward: {np.mean(r_save)}")
    log.info(f"Number of steps: {np.mean(step_save)}")
    log.info(f"Checkmates: {(r_save == 1).sum()}")
    log.info(f"Checkmate/Ration: {(r_save == 1).sum() / len(r_save)}")
    log.info(f"Amount of episode's needed: {len(r_save)}")
    log.info(f"--> LAST 100: Average reward: {np.mean(r_save[-100:])}")
    log.info(f"--> LAST 100: Number of steps: {np.mean(step_save[-100:])}")
    log.info(f"--> LAST 100: Checkmates: {(r_save[-100:] == 1).sum()}")


def generate_multi_plot(file_names, names, r_saves, step_saves):
    plt.subplots_adjust(wspace=1, hspace=0.3)

    plt.figure()
    plt.subplot(2, 1, 1)
    plot_curve_avg(names, r_saves, suffix=" $-$ last 100")
    plot_curve_avg(names, r_saves, last=500, suffix=" $-$ last 500")
    fig_setup('Checkmate')
    plt.title(f"Avg. Checkmates")

    plt.subplot(2, 1, 2)
    plot_curve_avg(names, step_saves, suffix=" $-$ last 100")
    plot_curve_avg(names, step_saves, last=500, suffix=" $-$ last 500")
    fig_setup('Step')
    plt.title(f"Avg. Steps")
    savefig(f"multi-plot-{'-'.join(file_names)}-learning_curve-100_500_avg.png")

    plt.figure()
    plt.subplot(2, 1, 1)
    plot_curve_avg(names, r_saves, last=500)
    fig_setup('Checkmate')
    plt.title(f"Avg. Checkmates $-$ last 500")

    plt.subplot(2, 1, 2)
    plot_curve_avg(names, step_saves, last=500)
    fig_setup('Step')
    plt.title(f"Avg. Steps $-$ last 500")
    savefig(f"multi-plot-{'-'.join(file_names)}-learning_curve-clean.png")

    plt.figure()
    plt.subplot(2, 1, 1)
    plot_curve_avg(names, r_saves, last=1000)
    fig_setup('Checkmate')
    plt.title(f"Avg. Checkmates $-$ last 1000")

    plt.subplot(2, 1, 2)
    plot_curve_avg(names, step_saves, last=1000)
    fig_setup('Step')
    plt.title(f"Avg. Steps $-$ last 1000")
    savefig(f"multi-plot-{'-'.join(file_names)}-learning_curve-super-clean.png")

    plt.figure()
    x = {}
    for idx in range(len(names)):
        moving_avg = generate_moving_avg(r_saves[idx], last=500)
        x[names[idx]] = moving_avg[-100:]
    df = pandas.DataFrame(x)
    df.boxplot()
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Avg. Checkmates for last 100")
    fig_setup('Avg. Checkmate', xlabel='')
    savefig(f"multi-plot-{'-'.join(file_names)}-avg-reward-box-plot.png")

    plt.figure()
    x = {}
    for idx in range(len(names)):
        x[names[idx]] = step_saves[idx][-100:]
    df = pandas.DataFrame(x)
    df.boxplot()
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Steps for last 100")
    fig_setup('Step', xlabel='')
    savefig(f"multi-plot-{'-'.join(file_names)}-steps-box-plot.png")


def generate_singe_plot(name, reward, move):
    evaluate_agent(name, reward)
    genrate_box_plots(name, reward)
    plt.close('all')


def savefig(filename):
    plt.tight_layout()
    plt.rcParams['figure.figsize'] = [6, 5]
    plt.savefig(config.model_data_file(filename), transparent="True", pad_inches=0)


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


def evaluate_agent(name, reward):
    moving_avg_500 = generate_moving_avg(reward, last=500)
    moving_avg_1000 = generate_moving_avg(reward, last=1000)

    plt.figure()
    plt.plot([i + 1 for i in range(500, len(reward))], moving_avg_500, label="Average last 500")
    plt.plot([i + 1 for i in range(1000, len(reward))], moving_avg_1000, label="Average last 1000")

    avg = np.average(reward)
    x = np.linspace(0, len(reward))
    plt.plot(x, [avg] * len(x), label="Average")
    plt.plot(x, [AIM] * len(x), label="Aim")
    fig_setup('Checkmate')
    savefig(f"{name}-learning_curve.png")

    plt.figure()
    plt.plot([i + 1 for i in range(1000, len(reward))], moving_avg_1000, label="Average last 1000")
    plt.plot(x, [avg] * len(x), label="Average")
    plt.plot(x, [AIM] * len(x), label="Aim")
    fig_setup('Checkmate')
    savefig(f"{name}-learning_curve_clean.png")


def fig_setup(ylabel, xlabel='Episode'):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()


def genrate_box_plots(name, reward):
    moving_avg = generate_moving_avg(reward, last=500)
    plt.figure()
    df = pandas.DataFrame({'Reward': moving_avg[-100:]})
    df.boxplot()
    savefig(f"{name}-boxplot_last_100.png")

    plt.figure()
    df = pandas.DataFrame({'Reward': moving_avg})
    df.boxplot()
    savefig(f"{name}-boxplot_all.png")


def generate_moving_avg(reward, last=100):
    return [np.average(reward[i - last:i]) for i in range(last, len(reward))]
