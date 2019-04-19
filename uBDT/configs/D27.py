from mods import config_path
config_path()

from D26 import uconfig

uconfig.features.train = [x for x in uconfig.features.train if x!="deltaphi"]
