from mods import config_path
config_path()

from D43 import uconfig

uconfig.features.train = [x for x in uconfig.features.train if not x in ["deltaphi","ptD","ptdrlog"]]
