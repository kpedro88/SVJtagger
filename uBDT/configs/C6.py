from mods import config_path
config_path()

from C2 import uconfig

uconfig.features.uniform = ["mt"]
uconfig.features.spectator = ["pt","eta"]
del uconfig.training.algorithms["bdt"]
