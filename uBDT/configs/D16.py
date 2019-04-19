from mods import config_path
config_path()

from D11 import uconfig

uconfig.hyper.power = 1.5
del uconfig.training.algorithms["bdt"]
