import math
import threading

import numpy as np
from flask import Flask, render_template
from turbo_flask import Turbo

from agents.q_table_agent import QTableAgent

app = Flask(__name__, static_folder='static', static_url_path='')
turbo = Turbo(app)

app_content = {'board': {'error': "No State yet"}, 'done': 0, 'info_text': f"Epoche (NOT STARTED YET)"}


@app.context_processor
def inject_board():
    return app_content


@app.route('/')
def index():
    return render_template('index.html')


@app.before_first_request
def before_first_request():
    threading.Thread(target=update_load).start()


def update_load():
    with app.app_context():
        def update_web(_, S, n, N_episodes, R_save, N_moves_save):
            if n % 10 == 0:
                global app_content
                app_content = {'board': calculate_location(S),
                               'epoche_string': f"{n}/{N_episodes}",
                               'average_reward': np.mean(R_save[(n - 100):n]),
                               'num_of_steps': np.mean(N_moves_save[(n - 100):n]),
                               'percentage': f"{n / N_episodes * 100}%",
                               'percentage_label': f"{math.ceil(n / N_episodes * 100)}%"
                               }
                turbo.push(turbo.replace(render_template('chess_board.html'), 'load'))

        QTableAgent(300000).run(update_web)


def calculate_location(S):
    board = np.array(S)
    board_location = {
        convert_location_to_letters(board, 1): 'wK',  # 1 = location of the King bK
        convert_location_to_letters(board, 2): 'wQ',  # 2 = location of the Queen wQ
        convert_location_to_letters(board, 3): 'bK',  # 3 = location fo the Enemy King wK
    }
    return board_location


def convert_location_to_letters(board, figure_id):
    match = np.where(board == figure_id)
    return f"{chr(97 + match[0][0])}{match[1][0] + 1}"
