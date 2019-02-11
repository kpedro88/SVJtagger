from mods import config_path
config_path()

from C2 import uconfig

uconfig.hyper.uloss = "exp"
del uconfig.training.algorithms["bdt"]
