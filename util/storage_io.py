import os

import dill as dill

from config import config


def get_filename(prefix: str):
    return config.model_data_file(f'{prefix}--model.sav')


def is_model_present(prefix):
    return os.path.isfile(get_filename(prefix))


def load_file(prefix):
    filename = get_filename(prefix)
    if not os.path.isfile(filename):
        print(f"File '{filename}' does not exist")
        return None
    with open(filename, "rb") as f:
        return dill.load(f)


def dump_file(content, prefix):
    filename = get_filename(prefix)
    with open(filename, 'wb') as f:
        dill.dump(content, f)
