from mods import config_path
config_path()

from D11 import uconfig

allsigs = [ "SVJ_mZprime-"+str(mZprime)+"_mDark-20_rinv-0.3_alpha-peak_MC2017" for mZprime in range(500,4600,100) ] \
        + [ "SVJ_mZprime-3000_mDark-"+str(mDark)+"_rinv-0.3_alpha-peak_MC2017" for mDark in [1,5]+range(10,110,10) ] \
        + [ "SVJ_mZprime-3000_mDark-20_rinv-"+str(rinv)+"_alpha-peak_MC2017" for rinv in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] ] \
        + [ "SVJ_mZprime-3000_mDark-20_rinv-0.3_alpha-"+alpha+"_MC2017" for alpha in ["peak","high","low"] ]
uconfig.dataset.signal = {"SVJ": list(sorted(set(allsigs)))}
#print len(uconfig.dataset.signal["SVJ"])
uconfig.features.spectator.extend(["mZprime","mDark","rinv","alpha"])
