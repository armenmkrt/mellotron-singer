from pprint import pprint

import yaml


class HParams(object):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            configs = yaml.load(f)
        self.__dict__.update(configs)

    def __setitem__(self, key, value): setattr(self, key, value)
    def __getitem__(self, key): return getattr(self, key)
    def __repr__(self): return pprint.pformat(self.__dict__)
