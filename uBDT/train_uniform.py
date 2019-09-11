from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# check arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-C","--config", dest="config", type=str, default="test1", help="config to provide parameters")
parser.add_argument("-s","--size", dest="trainTestSize", type=float, default=-1, help="size for test and train datasets (override config)")
parser.add_argument("-t","--threads", dest="threads", default=4, help="number of threads for parallel training")
parser.add_argument("-d","--dir", dest="dir", type=str, default="", help="directory for output files (required)")
parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", default=False, help="enable message printing")
args = parser.parse_args()

if len(args.dir)==0:
    parser.error("Required argument: --dir")

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
from collections import OrderedDict
import os

# restore warnings
reset_warn()

from mods import config_path
config_path()
uconfig = getattr(__import__(args.config.replace(".py",""),fromlist="uconfig"),"uconfig")
if args.trainTestSize > 0: uconfig.training.size = args.trainTestSize

# specify data
datasets = {
    "signal": uconfig.dataset.signal,
    "background": uconfig.dataset.background,
}
categories = {}
category_paths = {}
for dname in datasets:
    for cname in datasets[dname]:
        if cname in categories: raise ValueError("Repeated dataset category name: "+cname)
        if cname in datasets: raise ValueError("Reserved dataset name used as category: "+cname)
        categories[cname] = datasets[dname][cname]
        if isinstance(uconfig.dataset.path,dict): category_paths[cname] = uconfig.dataset.path[dname]
        else: category_paths[cname] = uconfig.dataset.path

# load and create dataframes
dfs = {}
ids = {}
wts = {}
for weight in uconfig.training.weights:
    wts[weight] = {}

# set up signal id criteria
if uconfig.training.signal_id_method=="two":
    if "index" not in uconfig.features.spectator: uconfig.features.spectator.append("index")
elif uconfig.training.signal_id_method=="isHV":
    if "isHV" not in uconfig.features.spectator: uconfig.features.spectator.append("isHV")
elif uconfig.training.signal_id_method=="isHVtwo":
    for x in ["index","isHV"]:
        if x not in uconfig.features.spectator: uconfig.features.spectator.append(x)
else:
    raise ValueError("Unknown signal_id_method: "+uconfig.training.signal_id_method)

for cname,clist in categories.iteritems():
    dfs[cname] = pd.DataFrame()
    for weight in uconfig.training.weights:
        wts[weight][cname] = pd.DataFrame()
    for sample in clist:
        if args.verbose: print sample
        f = up.open(category_paths[cname]+"tree_"+sample+".root")
        dfs[cname] = dfs[cname].append(f["tree"].pandas.df(uconfig.features.all_vars()))
        for weight in uconfig.training.weights:
            # make temporary df with all weight columns
            weightlist = f["tree"].pandas.df(uconfig.training.weights[weight]).abs() # ignore sign of weight
            if "procweight" in uconfig.training.weights[weight] and uconfig.training.signal_weight_method=="constant":
                weightlist["procweight"] = 1
            wprodtmp = pd.DataFrame()
            # easiest way to make a series into a dataframe
            wprodtmp["trainweight"] = weightlist.product(axis=1)
            wts[weight][cname] = wts[weight][cname].append(wprodtmp)

# apply signal id criteria
# make sure to mask weights as well
if uconfig.training.signal_id_method=="all":
    # nothing to do
    pass
if uconfig.training.signal_id_method=="two":
    for cname in categories:
        mask = (dfs[cname]["index"] < 2)
        dfs[cname] = dfs[cname][mask]
        for weight in uconfig.training.weights:
            wts[weight][cname] = wts[weight][cname][mask]
elif uconfig.training.signal_id_method=="isHV":
    dname = "signal"
    for cname in datasets[dname]:
        mask = (dfs[cname]["isHV"] > 0)
        dfs[cname] = dfs[cname][mask]
        for weight in uconfig.training.weights:
            wts[weight][cname] = wts[weight][cname][mask]
elif uconfig.training.signal_id_method=="isHVtwo":
    for dname in datasets:
        for cname in datasets[dname]:
            mask = (dfs[cname]["index"] < 2)
            if dname=="signal": mask = (dfs[cname]["index"] < 2) & (dfs[cname]["isHV"] > 0)
            dfs[cname] = dfs[cname][mask]
            for weight in uconfig.training.weights:
                wts[weight][cname] = wts[weight][cname][mask]
else:
    raise ValueError("Unknown signal_id_method: "+uconfig.training.signal_id_method)

# balance each sig and bkg category (make weights sum to 1)
# AFTER applying signal id criteria
for weight in uconfig.training.weights:
    for cname in categories:
        wts[weight][cname] /= np.sum(wts[weight][cname])

# combine categories into overall sig and bkg
for dname in datasets:
    dfs[dname] = pd.DataFrame()
    for weight in uconfig.training.weights:
        wts[weight][dname] = pd.DataFrame()
    for cname in datasets[dname]:
        dfs[dname] = dfs[dname].append(dfs[cname])
        for weight in uconfig.training.weights:
            wts[weight][dname] = wts[weight][dname].append(wts[weight][cname])

# balance combined sig and bkg
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
    if uconfig.hyper.uloss == "log":
        from mods import flat_log_loss
        flat_log_loss()
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

parallel_profile = None
if args.threads>1:
    parallel_profile = "threads-"+str(args.threads)
classifiers.fit(trainX, trainY, sample_weight=weights, parallel_profile=parallel_profile)

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
