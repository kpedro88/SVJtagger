from mods import config_path
config_path()

from D11 import uconfig

uconfig.dataset.signal = [ "SVJ_mZprime-"+str(mZprime)+"_mDark-20_rinv-0.3_alpha-peak_MC2017" for mZprime in range(500,4600,100) ]
uconfig.features.spectator.append("mZprime")
