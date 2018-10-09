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

# restore warnings
warnings.warn = warn_

# change default sizing
orig_init = AbstractPlot.__init__
def new_init(self):
    orig_init(self)
    self.figsize = (7,7)
AbstractPlot.__init__ = new_init

def saveplot(pname,plot,figsize=None):
    plot.plot(new_plot=True,figsize=figsize)
    fig = plt.gcf()
    fig.savefig(pname+".png",dpi=100)

# get reports
with open("train_uniform_reports.pkl",'rb') as infile:
    reports = pickle.load(infile)

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
