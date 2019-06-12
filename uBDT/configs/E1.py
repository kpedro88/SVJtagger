from uBDTConfig import uconfig

uconfig.dataset.path = "root://cmseos.fnal.gov//store/user/lpcsusyhad/SVJ2017/Run2ProductionV17/Skims/tree_dijetmtdetahadloosemf-train-flatsig/"
allsigs = [ "SVJ_mZprime-"+str(mZprime)+"_mDark-20_rinv-0.3_alpha-peak_MC2017" for mZprime in range(1500,4600,100) ] \
        + [ "SVJ_mZprime-3000_mDark-"+str(mDark)+"_rinv-0.3_alpha-peak_MC2017" for mDark in range(10,110,10) ] \
        + [ "SVJ_mZprime-3000_mDark-20_rinv-"+str(rinv)+"_alpha-peak_MC2017" for rinv in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] ] \
        + [ "SVJ_mZprime-3000_mDark-20_rinv-0.3_alpha-"+alpha+"_MC2017" for alpha in ["peak","high","low"] ]
uconfig.dataset.signal = {"SVJ": list(sorted(set(allsigs)))}
uconfig.dataset.background =  {
    "QCD": [
        "QCD_Pt_300to470_MC2017",
        "QCD_Pt_470to600_MC2017",
        "QCD_Pt_600to800_MC2017",
        "QCD_Pt_800to1000_MC2017",
        "QCD_Pt_1000to1400_MC2017",
        "QCD_Pt_1400to1800_MC2017",
        "QCD_Pt_1800to2400_MC2017",
        "QCD_Pt_2400to3200_MC2017",
    ],
    "ttbar": [
        "TTJets_MC2017",
        "TTJets_DiLept_MC2017",
        "TTJets_DiLept_genMET150_MC2017",
        "TTJets_SingleLeptFromT_MC2017",
        "TTJets_SingleLeptFromT_genMET150_MC2017",
        "TTJets_SingleLeptFromTbar_MC2017",
        "TTJets_SingleLeptFromTbar_genMET150_MC2017",
        "TTJets_HT600to800_MC2017",
        "TTJets_HT800to1200_MC2017",
        "TTJets_HT1200to2500_MC2017",
        "TTJets_HT2500toInf_MC2017",
    ],
}
uconfig.features.uniform = ["mt"]
uconfig.features.train = ["girth","tau21","tau32","msd","deltaphi","axisminor","axismajor","ptD","ecfN2b1","ecfN3b1","fChHad","fEle","fMu","fNeuHad","fPho"]
uconfig.features.spectator = ["pt","eta","mZprime","mDark","rinv","alpha","index"]
uconfig.training.size = 0.5
uconfig.training.signal_id_method = "isHVtwo"
uconfig.training.signal_weight_method = "default"
uconfig.training.weights = {
    "flat": ["flatweightZ30"],
    "proc": ["puweight","procweight"],
}
uconfig.training.algorithms = {
    "bdt": "flat",
    "ubdt": "proc",
}
uconfig.hyper.max_depth = 3
uconfig.hyper.n_estimators = 1000
uconfig.hyper.subsample = 0.6
uconfig.hyper.learning_rate = 0.1
uconfig.hyper.min_samples_leaf = 0.05
uconfig.hyper.fl_coefficient = 3
uconfig.hyper.power = 2.0
uconfig.hyper.uniform_label = 0
uconfig.hyper.n_bins = 20
uconfig.hyper.uloss = "log"
