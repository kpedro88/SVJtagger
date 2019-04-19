from mods import config_path
config_path()

from D11 import uconfig

uconfig.dataset.signal = [ "SVJ_mZprime-3000_mDark-20_rinv-"+str(rinv)+"_alpha-peak_MC2017" for rinv in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] ]
uconfig.features.spectator.append("rinv")
