from uBDTConfig import uconfig

uconfig.dataset.path = "root://cmseos.fnal.gov//store/user/lpcsusyhad/SVJ2017/Run2ProductionV16/Skims/tree_dijetmtdetahadmf-train-flatsig/"
uconfig.dataset.signal = {"SVJ": ["SVJ_mZprime-3000_mDark-20_rinv-0.3_alpha-peak_MC2017"]}
uconfig.dataset.background =  {"QCD": [
    "QCD_Pt_300to470_MC2017",
    "QCD_Pt_470to600_MC2017",
    "QCD_Pt_600to800_MC2017",
    "QCD_Pt_800to1000_MC2017",
    "QCD_Pt_1000to1400_MC2017",
    "QCD_Pt_1400to1800_MC2017",
    "QCD_Pt_1800to2400_MC2017",
    "QCD_Pt_2400to3200_MC2017",
]}
uconfig.features.uniform = ["mt"]
uconfig.features.train = ["mult","girth","tau21","tau32","msd","deltaphi","axisminor","axismajor","ptD","ecfN2b1","ecfN2b2","ecfN3b1","ecfN3b2","lean","ptdrlog","nSubjets"]
uconfig.features.spectator = ["pt","eta"]
uconfig.training.size = 0.5
uconfig.training.signal_id_method = "two"
uconfig.training.signal_weight_method = "default"
uconfig.training.weights = {
    "flat": "flatweightZ30",
    "proc": "procweight",
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
