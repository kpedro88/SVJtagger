# this is the only way to get rid of sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warn_ = warnings.warn
warnings.warn = warn

from rep.estimators import SklearnClassifier
from hep_ml import uboost, gradientboosting as ugb, losses
from rep.metaml import ClassifiersFactory
from rep.report.metrics import RocAuc
from hep_ml.metrics import BinBasedSDE, KnnBasedCvM
from rep.plotting import AbstractPlot
import matplotlib.pyplot as plt
import cPickle as pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict

# restore warnings
warnings.warn = warn_

# check arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-i","--input", dest="input", type=str, default="train_uniform_reports.pkl", help="name of .pkl file with reports")
parser.add_argument("-c","--classifiers", dest="classifiers", type=str, default=[], nargs='*', help="plot only for specified classifier(s) (space-separated)")
parser.add_argument("-s","--suffix", dest="suffix", type=str, default="", help="suffix for plots")
parser.add_argument("-f","--formats", dest="formats", type=str, default=["png"], nargs='*', help="print plots in specified format(s) (space-separated)")
args = parser.parse_args()

# change default sizing
orig_init = AbstractPlot.__init__
def new_init(self):
    orig_init(self)
    self.figsize = (7,7)
AbstractPlot.__init__ = new_init

def saveplot(pname,plot,figsize=None):
    plot.plot(new_plot=True,figsize=figsize)
    fig = plt.gcf()
    fname = pname
    if len(args.suffix)>0: fname += "_"+args.suffix
    for format in args.formats:
        fargs = {}
        if format=="png": fargs = {"dpi":100}
        elif format=="pdf": fargs = {"bbox_inches":"tight"}
        fig.savefig(fname+"."+format,**fargs)

# get reports
with open(args.input,'rb') as infile:
    reports = pickle.load(infile)

# check for subset of classifiers
if len(args.classifiers)>0:
    for rname,report in reports.iteritems():
        est_old = report.estimators
        pred_old = report.prediction
        est_new = ClassifiersFactory()
        pred_new = OrderedDict()
        for classifier in args.classifiers:
            if classifier in est_old:
                est_new[classifier] = est_old[classifier]
                pred_new[classifier] = pred_old[classifier]
            else:
                raise ValueError("Requested classifier "+classifier+" not found in report "+rname)
        report.estimators = est_new
        report.prediction = pred_new

# provides: uniform_features, train_features, spectators, all_vars
from features import *

# generate plots
labels = {0: "QCD", 1: "signal"}
plots = {}

plots["SpectatorEfficiencies"] = reports["test"].efficiencies(features=["pt","mt","eta"], bins=50, labels_dict=labels)
plots["CorrelationMatrix"] = reports["test"].features_correlation_matrix_by_class(labels_dict=labels)
plots["VariablePdfs"] = reports["test"].features_pdf(labels_dict=labels, bins=50, grid_columns=3)
plots["LearningCurveRocAuc"] = reports["test"].learning_curve(RocAuc(), steps=1)
plots["LearningCurveSDE"] = reports["test"].learning_curve(BinBasedSDE(uniform_features, uniform_label=1))
plots["LearningCurveCvM"] = reports["test"].learning_curve(KnnBasedCvM(uniform_features, uniform_label=1))
plots["PredictionTrain"] = reports["test"].prediction_pdf(labels_dict=labels, bins=50, plot_type='bar')
plots["PredictionTest"] = reports["train"].prediction_pdf(labels_dict=labels, bins=50, plot_type='bar')
plots["RocCurve"] = reports["test"].roc(physics_notion=True)

# need to reset classifier features to only trained features (eliminating spectators)
for name, estimator in reports["test"].estimators.items():
    estimator.features = train_features
plots["FeatureImportance"] = reports["test"].feature_importance()

# plot w/ matplotlib because most things not supported for ROOT/TMVA style
for pname,plot in sorted(plots.iteritems()):
    # separate these
    if pname=="CorrelationMatrix":
        for i,iplot in enumerate(plot.plots):
            saveplot(pname+"_"+labels[i],iplot,figsize=(7,7))
    else:
        saveplot(pname,plot)
