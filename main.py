import numpy as np

from agents.q_table_agent import QTableAgent, QTableAgentCustomReward, QTableAgentCustomReward2
from util.logger import log
from util.plotting import generate_multi_plot, \
    generate_singe_plot
from util.storage_io import dump_file, load_file, is_model_present

intern_output_nr = 500


def print_to_console(agent, _, n, N_episodes, R_save, N_moves_save):
    if n % intern_output_nr == 0 and n > 0:
        dump_file(agent, agent.clean_name())
        log.info(f"Epoche ({n}/{N_episodes})")
        log.info(f"{agent.NAME}, Average reward: {np.mean(R_save[(n - intern_output_nr):n])} "
                 + f"Number of steps: {np.mean(N_moves_save[(n - intern_output_nr):n])}")


if __name__ == '__main__':
    names = []
    rewards = []
    moves = []
    N_episodes = 50000

    for agent_class in [QTableAgent, QTableAgentCustomReward, QTableAgentCustomReward2]:
        agent = agent_class(N_episodes)
        model_filename = f"{agent.clean_name()}_model_content"

        if is_model_present(model_filename):
            name, reward, move = load_file(model_filename)
        else:
            name, reward, move = agent.run(print_to_console)
            dump_file([name, reward, move], model_filename)
        log.info(f'Finished with {len(reward)} Epochs')
        log.info(f'Generating plots for: {name}...')
        name = agent.clean_name()
        generate_singe_plot(name, reward, move)

        names.append(name)
        rewards.append(reward)
        moves.append(move)
    generate_multi_plot(names, rewards, moves)
