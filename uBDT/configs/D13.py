from mods import config_path
config_path()

from D11 import uconfig

uconfig.dataset.signal = [ "SVJ_mZprime-3000_mDark-"+str(mDark)+"_rinv-0.3_alpha-peak_MC2017" for mDark in [1,5]+range(10,110,10) ]
uconfig.features.spectator.append("mDark")
