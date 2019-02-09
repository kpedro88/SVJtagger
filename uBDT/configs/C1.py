from uBDTConfig import uconfig

uconfig.dataset.path = "root://cmseos.fnal.gov//store/user/lpcsusyhad/SVJ2017/Run2ProductionV16/Skims/tree_dijetmtdetahadmf-train-flatsig/"
uconfig.dataset.signal = ["SVJ_mZprime-3000_mDark-20_rinv-0.3_alpha-peak_MC2017"]
uconfig.dataset.background =  [
    "QCD_Pt_300to470_MC2017",
    "QCD_Pt_470to600_MC2017",
    "QCD_Pt_600to800_MC2017",
    "QCD_Pt_800to1000_MC2017",
    "QCD_Pt_1000to1400_MC2017",
    "QCD_Pt_1400to1800_MC2017",
    "QCD_Pt_1800to2400_MC2017",
    "QCD_Pt_2400to3200_MC2017",
]
uconfig.features.uniform = ["pt"]
uconfig.features.train = ["mult","girth","tau21","tau32","msd","deltaphi","axisminor"]
uconfig.features.spectator = ["mt","eta"]
uconfig.training.size = 0.5
uconfig.training.signal_id_method = "two"
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
uconfig.hyper.learning_rate = 1.0
uconfig.hyper.min_samples_leaf = 0.05
uconfig.hyper.fl_coefficient = 3
uconfig.hyper.power = 1.3
uconfig.hyper.uniform_label = 1
uconfig.hyper.n_bins = 20
