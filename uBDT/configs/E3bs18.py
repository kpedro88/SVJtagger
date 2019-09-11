from uBDTConfig import uconfig

from E3 import uconfig

uconfig.dataset.path = {
    "background": "root://cmseos.fnal.gov//store/user/lpcsusyhad/SVJ2017/Run2ProductionV17/Skims/tree_dijetmtdetahadloosemf-train-flatsig/",
    "signal": "root://cmseos.fnal.gov//store/user/lpcsusyhad/SVJ2017/Run2ProductionV17/Skims/scan/tree_dijetmtdetahadloosemf-train-flatsig/",
}

allsigs = [ "SVJ_"+str(mZprime)+"_"+str(mDark)+"_"+str(rinv)+"_peak_MC2018" for mZprime in range(1500,5200,200) for (mDark,rinv) in [(20,0.3),(30,0.3),(20,0.4)] ] \
        + [ "SVJ_"+str(mZprime)+"_"+str(mDark)+"_0.3_peak_MC2018" for mDark in range(10,110,10) for mZprime in [2700,2900,3100,3300] ] \
        + [ "SVJ_"+str(mZprime)+"_20_"+str(rinv)+"_peak_MC2018" for rinv in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] for mZprime in [2700,2900,3100,3300] ] \
        + [ "SVJ_"+str(mZprime)+"_20_0.3_"+alpha+"_MC2018" for alpha in ["peak","high","low"]  for mZprime in [2700,2900,3100,3300] ]
uconfig.dataset.signal = {"SVJ": list(sorted(set(allsigs)))}

for bkg in uconfig.dataset.background:
    uconfig.dataset.background[bkg] = [x.replace("MC2018","MC2018") for x in uconfig.dataset.background[bkg] if not "genMET150" in x]
