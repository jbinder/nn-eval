import importlib

from common.tools import convert_to_snake_case


def get_normalizer(normalizer, data):
    try:
        file_name = convert_to_snake_case(normalizer)
        module = importlib.import_module("normalizer." + file_name + "_normalizer")
        if normalizer == "SklearnStandard":
            normalizer = getattr(module, normalizer + "Normalizer")(data)
        else:
            normalizer = getattr(module, normalizer + "Normalizer")()
        return normalizer
    except ModuleNotFoundError:
        raise Exception(f"Normalizer not supported: {normalizer}")
