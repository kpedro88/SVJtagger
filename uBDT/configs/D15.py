from mods import config_path
config_path()

from D11 import uconfig

uconfig.dataset.signal = [ "SVJ_mZprime-3000_mDark-20_rinv-0.3_alpha-"+alpha+"_MC2017" for alpha in ["peak","high","low"] ]
uconfig.features.spectator.append("alpha")
