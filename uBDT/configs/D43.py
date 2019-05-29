from mods import config_path
config_path()

from D39 import uconfig

uconfig.dataset.signal = {"ttbar": [
    "TTJets",
    "TTJets_DiLept",
    "TTJets_DiLept_genMET150",
    "TTJets_SingleLeptFromT",
    "TTJets_SingleLeptFromT_genMET150",
    "TTJets_SingleLeptFromTbar",
    "TTJets_SingleLeptFromTbar_genMET150",
    "TTJets_HT600to800",
    "TTJets_HT800to1200",
    "TTJets_HT1200to2500",
    "TTJets_HT2500toInf",
]}
uconfig.training.signal_id_method = "two"
uconfig.training.weights["flat"] = "flatweightttbar"
