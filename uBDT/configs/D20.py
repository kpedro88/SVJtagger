from mods import config_path
config_path()

from D11 import uconfig

uconfig.hyper.fl_coefficient = 10
del uconfig.training.algorithms["bdt"]
