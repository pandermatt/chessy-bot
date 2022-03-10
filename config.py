import os
from os.path import dirname, abspath, join, exists


def get_or_create(dir_path):
    if not exists(dir_path):
        os.makedirs(dir_path)
        print("Creating: " + dir_path)
    return dir_path


class Config:
    SAVE_TO_FILE = True

    __root_dir = dirname(abspath(__file__))

    def __data_dir(self):
        return get_or_create(abspath(join(self.__root_dir, 'data')))

    def model_data_file(self, file_name):
        return join(self.__data_dir(), file_name)


config = Config()
