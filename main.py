import numpy as np

from agents.chessy_agent import QLearningChessyAgent, QLearningChessyAgentLeakyReLU, \
    QLearningChessyAgentRMSProp, QLearningChessyAgentRMSPropLeakyReLU, SarsaChessyAgentOneHidden, SarsaChessyAgent, \
    SarsaChessyAgentThreeHidden, SarsaChessyAgentAdam, SarsaChessyAgentRMSProp, QLearningChessyAgentHighReward, \
    SarsaChessyAgentHighReward, QLearningChessyAgentCustomReward, QLearningChessyAgentCustomReward2, \
    SarsaChessyAgentCustomReward, SarsaChessyAgentCustomReward2, DoubleQLearningChessyAgent, DoubleSarsaChessyAgent, \
    SarsaChessyAgentSigmoid, SarsaChessyAgentLeakyReLU
from agents.q_table_agent import QTableAgent, QTableAgentCustomReward2, QTableAgentCustomReward
from util.logger import log
from util.plotting import generate_multi_plot, \
    generate_singe_plot, print_stats
from util.storage_io import dump_file, load_file, is_model_present

intern_output_nr = 500


def print_to_console(agent, _, n, N_episodes, R_save, N_moves_save):
    if n % intern_output_nr == 0 and n > 0:
        dump_file(agent, agent.clean_name())
        log.info(f"Episode ({n}/{N_episodes})")
        log.info(f"{agent.NAME}, Average reward: {np.mean(R_save[(n - intern_output_nr):n])} "
                 + f"Number of steps: {np.mean(N_moves_save[(n - intern_output_nr):n])}")


def run_and_compare(agent_class_list):
    np.random.seed(42)
    N_episodes = 100000

    file_names = []
    names = []
    rewards = []
    moves = []

    for agent_class in agent_class_list:
        agent = agent_class(N_episodes)
        model_filename = f"{agent.clean_name()}_model_content"

        if is_model_present(model_filename):
            _, reward, move = load_file(model_filename)
        else:
            name, reward, move = agent.run(print_to_console)
            dump_file([name, reward, move], model_filename)

        name = agent.NAME
        log.info(f'Finished with {len(reward)} Episodes')
        log.info(f'Generating plots for: {name}...')
        file_name = agent.clean_name()

        reward = np.divide(reward, agent.env.reward_checkmate)
        print_stats(name, reward, move)
        generate_singe_plot(file_name, reward, move)

        file_names.append(file_name)
        names.append(name)
        rewards.append(reward)
        moves.append(move)
    generate_multi_plot(file_names, names, rewards, moves)


if __name__ == '__main__':
    # compare activation functions
    run_and_compare([SarsaChessyAgent,
                     SarsaChessyAgentSigmoid,
                     SarsaChessyAgentLeakyReLU])

    # compare deep nets
    run_and_compare([SarsaChessyAgent,
                     QLearningChessyAgent,
                     QTableAgent])

    run_and_compare([SarsaChessyAgent,
                     QLearningChessyAgent,
                     DoubleSarsaChessyAgent,
                     DoubleQLearningChessyAgent])

    # compare q_tables
    run_and_compare([QTableAgent,
                     QTableAgentCustomReward,
                     QTableAgentCustomReward2])

    # compare negative reward
    run_and_compare([SarsaChessyAgent,
                     SarsaChessyAgentCustomReward,
                     SarsaChessyAgentCustomReward2])

    # compare negative reward
    run_and_compare([QLearningChessyAgent,
                     QLearningChessyAgentCustomReward,
                     QLearningChessyAgentCustomReward2])

    # compare high reward
    run_and_compare([SarsaChessyAgent,
                     SarsaChessyAgentHighReward,
                     QLearningChessyAgent,
                     QLearningChessyAgentHighReward])

    # compare optimizers
    run_and_compare([SarsaChessyAgent,
                     SarsaChessyAgentAdam,
                     SarsaChessyAgentRMSProp])

    # compare hidden layer_sizes
    run_and_compare([SarsaChessyAgentOneHidden, SarsaChessyAgent, SarsaChessyAgentThreeHidden])

    run_and_compare([QLearningChessyAgent,
                     QLearningChessyAgentLeakyReLU,
                     QLearningChessyAgentRMSProp,
                     QLearningChessyAgentRMSPropLeakyReLU])
