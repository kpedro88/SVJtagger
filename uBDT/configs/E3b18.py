from uBDTConfig import uconfig

from E3 import uconfig

for bkg in uconfig.dataset.background:
    uconfig.dataset.background[bkg] = [x.replace("MC2017","MC2018") for x in uconfig.dataset.background[bkg] if not "genMET150" in x]
