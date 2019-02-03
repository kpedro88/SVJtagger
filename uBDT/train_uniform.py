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
import os

# restore warnings
reset_warn()

# check arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-C","--config", dest="config", type=str, default="test1", help="config to provide parameters")
parser.add_argument("-t","--train-test-size", dest="trainTestSize", type=float, default=-1, help="size for test and train datasets (override config)")
parser.add_argument("-d","--dir", dest="dir", type=str, default="", help="directory for output files (required)")
parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", default=False, help="enable message printing")
args = parser.parse_args()

if len(args.dir)==0:
    parser.error("Required argument: --dir")

from mods import config_path
config_path()
uconfig = getattr(__import__(args.config.replace(".py",""),fromlist="uconfig"),"uconfig")
if args.trainTestSize > 0: uconfig.training.size = args.trainTestSize

# specify data
datasets = {
    "signal": uconfig.dataset.signal,
    "background": uconfig.dataset.background,
}

# load and create dataframes
dfs = {}
ids = {}
wts = {}
for weight in uconfig.training.weights:
    wts[weight] = {}

# set up signal id criteria
if uconfig.training.signal_id_method=="two":
    uconfig.features.spectator.append("index")
elif uconfig.training.signal_id_method=="isHV":
    uconfig.features.spectator.append("isHV")

for dname,dlist in datasets.iteritems():
    dfs[dname] = pd.DataFrame()
    for weight in uconfig.training.weights:
        wts[weight][dname] = pd.DataFrame()
    for sample in dlist:
        if args.verbose: print sample
        f = up.open(uconfig.dataset.path+"tree_"+sample+".root")
        dfs[dname] = dfs[dname].append(f["tree"].pandas.df(uconfig.features.all_vars()))
        for weight in uconfig.training.weights:
            wts[weight][dname] = wts[weight][dname].append(f["tree"].pandas.df([uconfig.training.weights[weight]]))

# apply signal id criteria
# make sure to mask weights as well
if uconfig.training.signal_id_method=="all":
    # nothing to do
    pass
if uconfig.training.signal_id_method=="two":
    for dname in datasets:
        mask = (dfs[dname]["index"] < 2)
        dfs[dname] = dfs[dname][mask]
        for weight in uconfig.training.weights:
            wts[weight][dname] = wts[weight][dname][mask]
elif uconfig.training.signal_id_method=="isHV":
    dname = "signal"
    mask = (dfs[dname]["isHV"])
    dfs[dname] = dfs[dname][mask]
    for weight in uconfig.training.weights:
        wts[weight][dname] = wts[weight][dname][mask]
else:
    raise ValueError("Unknown signal_id_method: "+uconfig.training.signal_id_method)

# balance sig vs. bkg (make weights sum to 1)
# AFTER applying signal id criteria
for weight in uconfig.training.weights:
    for dname in datasets:
        wts[weight][dname] /= np.sum(wts[weight][dname])

# classifications
ids["signal"] = np.ones(len(dfs["signal"]))
ids["background"] = np.zeros(len(dfs["background"]))

if args.verbose: fprint("Loaded data")

# split dataset into train and test
X = pd.concat([dfs["signal"],dfs["background"]])
Y = np.concatenate((ids["signal"],ids["background"]))
W = {}
for weight in uconfig.training.weights:
    W[weight] = pd.concat([wts[weight]["signal"],wts[weight]["background"]]).values
    W[weight].shape = (W[weight].size)
train_test_data = train_test_split(
    X, Y, *[v for k,v in sorted(W.iteritems())],
    test_size=uconfig.training.size,
    train_size=uconfig.training.size,
    random_state=42
)
train_data = train_test_data[::2]
test_data = train_test_data[1::2]
trainX = train_data[0]
testX = test_data[0]
trainY = train_data[1]
testY = test_data[1]
trainW = {}
testW = {}
for iw,weight in enumerate(sorted(W)):
    trainW[weight] = train_data[iw+2]
    testW[weight] = test_data[iw+2]

if args.verbose: fprint("Split data into train_size="+str(uconfig.training.size)+", test_size="+str(uconfig.training.size))

# create classifiers
classifiers = ClassifiersFactory()
weights = OrderedDict()

# standard bdt
if "bdt" in uconfig.training.algorithms:
    base_grad = GradientBoostingClassifier(
        max_depth=uconfig.hyper.max_depth,
        n_estimators=uconfig.hyper.n_estimators,
        subsample=uconfig.hyper.subsample,
        learning_rate=uconfig.hyper.learning_rate,
        min_samples_leaf=uconfig.hyper.min_samples_leaf,
    )
    classifiers["bdt"] = SklearnClassifier(base_grad, features=uconfig.features.train)
    weights["bdt"] = trainW[uconfig.training.algorithms["bdt"]]

# uniform bdt
if "ubdt" in uconfig.training.algorithms:
    flatnessloss = ugb.BinFlatnessLossFunction(
        uconfig.features.uniform,
        fl_coefficient=uconfig.hyper.fl_coefficient,
        power=uconfig.hyper.power,
        uniform_label=uconfig.hyper.uniform_label,
        n_bins=uconfig.hyper.n_bins,
    )
    ugbFL = ugb.UGradientBoostingClassifier(
        loss=flatnessloss,
        max_depth=uconfig.hyper.max_depth,
        n_estimators=uconfig.hyper.n_estimators,
        subsample=uconfig.hyper.subsample,
        learning_rate=uconfig.hyper.learning_rate,
        min_samples_leaf=uconfig.hyper.min_samples_leaf,
        train_features=uconfig.features.train,
    )
    # do not use features=uconfig.features.train here, otherwise it won't know about features.uniform
    classifiers["ubdt"] = SklearnClassifier(ugbFL)
    weights["ubdt"] = trainW[uconfig.training.algorithms["ubdt"]]

if args.verbose: fprint("Start training")

from mods import fit_separate_weights
fit_separate_weights()

classifiers.fit(trainX, trainY, sample_weight=weights, parallel_profile='threads-4')

if args.verbose: fprint("Finish training")

# saving results
import cPickle as pickle
from skTMVA import skTMVA

if args.verbose: fprint("Start saving")

prefix = args.dir
if prefix[-1]!="/": prefix += "/"
if not os.path.exists(args.dir):
    os.makedirs(args.dir)

cname = prefix+"train_uniform_classifiers.pkl"
wname_bdt = prefix+"TMVA_bdt_weights.xml"
wname_ubdt = prefix+"TMVA_ubdt_weights.xml"
rname = prefix+"train_uniform_reports.pkl"

# save classifiers to pkl file
with open(cname, 'wb') as outfile:
    pickle.dump(classifiers, outfile)

# save in TMVA format
tmva_vars = [(f,'F') for f in uconfig.features.train]

if "bdt" in uconfig.training.algorithms:
    skTMVA.convert_bdt__Grad(classifiers["bdt"],tmva_vars,wname_bdt)

# make UGradientBoostingClassifier compatible w/ sklearn GradientBoostingClassifier
if "ubdt" in uconfig.training.algorithms:
    from mods import uGB_to_GB
    uGB_to_GB(classifiers["ubdt"])
    skTMVA.convert_bdt__Grad(classifiers["ubdt"],tmva_vars,wname_ubdt)

# save reports
reports = {}
# have to evaluate with all sets of weights because of report structure
for weight in sorted(W):
    reports["train"+weight] = classifiers.test_on(trainX, trainY, sample_weight=trainW[weight])
    reports["test"+weight] = classifiers.test_on(testX, testY, sample_weight=testW[weight])
with open(rname, 'wb') as outfile:
    pickle.dump(reports, outfile)

if args.verbose: fprint("Finish saving")
