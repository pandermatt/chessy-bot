import threading

from flask import Flask, render_template
from turbo_flask import Turbo

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
    from chessy_agent import ChessyAgent

    with app.app_context():
        def update_web(params):
            global app_content
            app_content = params
            turbo.push(turbo.replace(render_template('chess_board.html'), 'load'))

        ChessyAgent().run(update_web)
