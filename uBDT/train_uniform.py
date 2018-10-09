# this is the only way to get rid of sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warn_ = warnings.warn
warnings.warn = warn

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

# restore warnings
warnings.warn = warn_

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

# define features
uniform_features = ["pt"]
train_features = ["mult","axisminor","girth","tau21","tau32","msd","deltaphi"]
spectators = ["mt","eta"]
all_vars = uniform_features+train_features+spectators

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

# split dataset into train and test
X = pd.concat([dfs["signal"],dfs["background"]])
Y = np.concatenate((cls["signal"],cls["background"]))
W = pd.concat((wts["signal"],wts["background"])).values
trainX, testX, trainY, testY, trainW, testW = train_test_split(X, Y, W, test_size=0.01, train_size=0.01, random_state=42)

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

flatnessloss = ugb.KnnFlatnessLossFunction(uniform_features, fl_coefficient=3., power=1.3, uniform_label=1)
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

classifiers.fit(trainX, trainY, sample_weight=trainW, parallel_profile='threads-4')

# saving results
import cPickle as pickle
from skTMVA import skTMVA

# save classifiers to pkl file
with open('train_uniform_classifiers.pkl', 'wb') as outfile:
	pickle.dump(classifiers['GradBoost'], outfile)
	pickle.dump(classifiers['uGBFL'], outfile)

# save in TMVA format
tmva_vars = [(f,'F') for f in train_features]
#skTMVA.convert_bdt_sklearn_tmva(classifiers['GradBoost'],tmva_vars,'TMVA_GradBoost_weights.xml')
#skTMVA.convert_bdt_sklearn_tmva(classifiers['uGBFL'],tmva_vars,'TMVA_uGBFL_weights.xml')
skTMVA.convert_bdt__Grad(classifiers['GradBoost'],tmva_vars,'TMVA_GradBoost_weights.xml')

# make UGradientBoostingClassifier compatible w/ sklearn GradientBoostingClassifier
classifiers['uGBFL'].loss_ = classifiers['uGBFL'].loss
classifiers['uGBFL'].loss_.K = 1
classifiers['uGBFL'].estimators_ = np.empty((classifiers['uGBFL'].n_estimators, classifiers['uGBFL'].loss_.K), dtype=np.object)
for i,est in enumerate(classifiers['uGBFL'].estimators):
    classifiers['uGBFL'].estimators_[i] = est[0]
    classifiers['uGBFL'].estimators_[i][0].leaf_values = est[1]
skTMVA.convert_bdt__Grad(classifiers['uGBFL'],tmva_vars,'TMVA_uGBFL_weights.xml')

# save reports
reports = {}
reports["train"] = classifiers.test_on(trainX, trainY)
reports["test"] = classifiers.test_on(testX, testY)
with open('train_uniform_reports.pkl', 'wb') as outfile:
	pickle.dump(reports, outfile)
