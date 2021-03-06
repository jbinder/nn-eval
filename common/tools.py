import re
from os.path import dirname, abspath


def convert_to_snake_case(name):
    """ see https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def get_root_dir():
    return dirname(dirname(abspath(__file__)))
