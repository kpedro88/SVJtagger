# get rid of sklearn warnings
from mods import suppress_warn, reset_warn, fprint
suppress_warn()

import numpy as np
import uproot as up
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from rep.estimators import SklearnClassifier
from hep_ml.commonutils import train_test_split
from hep_ml import uboost, gradientboosting as ugb, losses
from rep.metaml import ClassifiersFactory
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict

# restore warnings
reset_warn()

# check arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-t","--train-test-size", dest="trainTestSize", type=float, default=0.5, help="size for test and train datasets")
parser.add_argument("-s","--suffix", dest="suffix", type=str, default="", help="suffix for output files")
parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", default=False, help="enable message printing")
args = parser.parse_args()

# specify data
path = "root://cmseos.fnal.gov//store/user/lpcsusyhad/SVJ2017/Run2ProductionV14/Skims/tree_dijetmthadloose-train-flatsig/"
datasets = {
    "signal": [
        "SVJ2_mZprime-3000_mDark-20_rinv-0.3_alpha-0.2",
    ],
    "background": [
        "QCD_Pt_80to120",
        "QCD_Pt_120to170",
        "QCD_Pt_170to300",
        "QCD_Pt_300to470",
        "QCD_Pt_470to600",
        "QCD_Pt_600to800",
        "QCD_Pt_800to1000",
        "QCD_Pt_1000to1400",
        "QCD_Pt_1400to1800",
        "QCD_Pt_1800to2400",
        "QCD_Pt_2400to3200",
    ],
}

# provides: uniform_features, train_features, spectators, all_vars
from features import *

# load and create dataframes
dfs = {}
ids = {}
fwts = {}
pwts = {}

for dname,dlist in datasets.iteritems():
    dfs[dname] = pd.DataFrame()
    fwts[dname] = pd.DataFrame()
    pwts[dname] = pd.DataFrame()
    for sample in dlist:
        f = up.open(path+"tree_"+sample+".root")
        dfs[dname] = dfs[dname].append(f["tree"].pandas.df(all_vars))
        # apply pt flattening weights or proc weights
        fwts[dname] = fwts[dname].append(f["tree"].pandas.df(["flatweight"]))
        pwts[dname] = pwts[dname].append(f["tree"].pandas.df(["weight"]))

# balance sig vs. bkg (make weights sum to 1)
fwts["signal"] /= np.sum(fwts["signal"])
pwts["signal"] /= np.sum(pwts["signal"])
fwts["background"] /= np.sum(fwts["background"])
pwts["background"] /= np.sum(pwts["background"])

# classifications
ids["signal"] = np.ones(len(dfs["signal"]))
ids["background"] = np.zeros(len(dfs["background"]))

if args.verbose: fprint("Loaded data")

# split dataset into train and test
X = pd.concat([dfs["signal"],dfs["background"]])
Y = np.concatenate((ids["signal"],ids["background"]))
FW = pd.concat([fwts["signal"],fwts["background"]]).values
FW.shape = (FW.size)
PW = pd.concat([pwts["signal"],pwts["background"]]).values
PW.shape = (PW.size)
trainX, testX, trainY, testY, trainFW, testFW, trainPW, testPW = train_test_split(
	X, Y, FW, PW, 
	test_size=args.trainTestSize,
	train_size=args.trainTestSize,
	random_state=42
)

if args.verbose: fprint("Split data into train_size="+str(args.trainTestSize)+", test_size="+str(args.trainTestSize))

# create classifiers
classifiers = ClassifiersFactory()
weights = OrderedDict()

base_grad = GradientBoostingClassifier(
    max_depth=3,
    n_estimators=1000,
    subsample=0.6,
    learning_rate=1.0,
    min_samples_leaf=0.05,
)
classifiers['GradBoost'] = SklearnClassifier(base_grad, features=train_features)
weights['GradBoost'] = trainFW

# uniform in signal pt
flatnessloss = ugb.BinFlatnessLossFunction(uniform_features, fl_coefficient=3., power=1.3, uniform_label=1, n_bins=20)
ugbFL = ugb.UGradientBoostingClassifier(
    loss=flatnessloss,
    max_depth=3,
    n_estimators=1000,
    subsample=0.6,
    learning_rate=1.0,
    min_samples_leaf=0.05,
    train_features=train_features,
)
# do not use features=train_features here, otherwise it won't know about uniform_features
classifiers['uGBFL'] = SklearnClassifier(ugbFL)
weights['uGBFL'] = trainPW

if args.verbose: fprint("Start training")

from mods import fit_separate_weights
fit_separate_weights()

classifiers.fit(trainX, trainY, sample_weight=weights, parallel_profile='threads-4')

if args.verbose: fprint("Finish training")

# saving results
import cPickle as pickle
from skTMVA import skTMVA

if args.verbose: fprint("Start saving")

if len(args.suffix)>0: args.suffix = "_"+args.suffix
cname = "train_uniform_classifiers"+args.suffix+".pkl"
wname_GB = "TMVA_GradBoost_weights"+args.suffix+".xml"
wname_uGB = "TMVA_uGBFL_weights"+args.suffix+".xml"
rname = "train_uniform_reports"+args.suffix+".pkl"

# save classifiers to pkl file
with open(cname, 'wb') as outfile:
    pickle.dump(classifiers, outfile)

# save in TMVA format
tmva_vars = [(f,'F') for f in train_features]
skTMVA.convert_bdt__Grad(classifiers['GradBoost'],tmva_vars,wname_GB)

# make UGradientBoostingClassifier compatible w/ sklearn GradientBoostingClassifier
from mods import uGB_to_GB
uGB_to_GB(classifiers['uGBFL'])
skTMVA.convert_bdt__Grad(classifiers['uGBFL'],tmva_vars,wname_uGB)

# save reports
reports = {}
# have to train with both sets of weights because of report structure
reports["trainF"] = classifiers.test_on(trainX, trainY, sample_weight=trainFW)
reports["testF"] = classifiers.test_on(testX, testY, sample_weight=testFW)
reports["trainP"] = classifiers.test_on(trainX, trainY, sample_weight=trainPW)
reports["testP"] = classifiers.test_on(testX, testY, sample_weight=testPW)
with open(rname, 'wb') as outfile:
    pickle.dump(reports, outfile)

if args.verbose: fprint("Finish saving")
