# this is the only way to get rid of sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warn_ = warnings.warn
warnings.warn = warn

import sys
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

# restore warnings
warnings.warn = warn_

# make status messages useful
def fprint(msg):
    print(msg)
    sys.stdout.flush()

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
cls = {}
wts = {}

for dname,dlist in datasets.iteritems():
    dfs[dname] = pd.DataFrame()
    wts[dname] = pd.DataFrame()
    for sample in dlist:
        f = up.open(path+"tree_"+sample+".root")
        dfs[dname] = dfs[dname].append(f["tree"].pandas.df(all_vars))
        # apply pt flattening weights
        wts[dname] = wts[dname].append(f["tree"].pandas.df(["flatweight"]))

# classifications
cls["signal"] = np.ones(len(dfs["signal"]))
cls["background"] = np.zeros(len(dfs["background"]))

if args.verbose: fprint("Loaded data")

# split dataset into train and test
X = pd.concat([dfs["signal"],dfs["background"]])
Y = np.concatenate((cls["signal"],cls["background"]))
W = pd.concat((wts["signal"],wts["background"])).values
trainX, testX, trainY, testY, trainW, testW = train_test_split(X, Y, W, test_size=args.trainTestSize, train_size=args.trainTestSize, random_state=42)

if args.verbose: fprint("Split data into train_size="+str(args.trainTestSize)+", test_size="+str(args.trainTestSize))

# create classifiers
classifiers = ClassifiersFactory()
base_grad = GradientBoostingClassifier(
    max_depth=3,
    n_estimators=1000,
    subsample=0.6,
    learning_rate=1.0,
    min_samples_leaf=0.05,
)
classifiers['GradBoost'] = SklearnClassifier(base_grad, features=train_features)

# uniform in signal and background
flatnessloss = ugb.KnnFlatnessLossFunction(uniform_features, fl_coefficient=3., power=1.3, uniform_label=[0,1])
ugbFL = ugb.UGradientBoostingClassifier(
    loss=flatnessloss,
    max_depth=3,
    n_estimators=1000,
    subsample=0.6,
    learning_rate=1.0,
    min_samples_leaf=0.05,
    train_features=train_features,
)
classifiers['uGBFL'] = SklearnClassifier(ugbFL)

if args.verbose: fprint("Start training")

classifiers.fit(trainX, trainY, sample_weight=trainW, parallel_profile='threads-4')

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
#skTMVA.convert_bdt_sklearn_tmva(classifiers['GradBoost'],tmva_vars,'TMVA_GradBoost_weights.xml')
#skTMVA.convert_bdt_sklearn_tmva(classifiers['uGBFL'],tmva_vars,'TMVA_uGBFL_weights.xml')
skTMVA.convert_bdt__Grad(classifiers['GradBoost'],tmva_vars,wname_GB)

# make UGradientBoostingClassifier compatible w/ sklearn GradientBoostingClassifier
classifiers['uGBFL'].loss_ = classifiers['uGBFL'].loss
classifiers['uGBFL'].loss_.K = 1
classifiers['uGBFL'].estimators_ = np.empty((classifiers['uGBFL'].n_estimators, classifiers['uGBFL'].loss_.K), dtype=np.object)
for i,est in enumerate(classifiers['uGBFL'].estimators):
    classifiers['uGBFL'].estimators_[i] = est[0]
    classifiers['uGBFL'].estimators_[i][0].leaf_values = est[1]
skTMVA.convert_bdt__Grad(classifiers['uGBFL'],tmva_vars,wname_uGB)

# save reports
reports = {}
reports["train"] = classifiers.test_on(trainX, trainY, sample_weight=trainW)
reports["test"] = classifiers.test_on(testX, testY, sample_weight=testW)
with open(rname, 'wb') as outfile:
    pickle.dump(reports, outfile)

if args.verbose: fprint("Finish saving")
