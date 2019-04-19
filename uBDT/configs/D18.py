from mods import config_path
config_path()

from D11 import uconfig

uconfig.hyper.power = 5.0
del uconfig.training.algorithms["bdt"]
